"""
Metric conversion.

This module is used to convert dbt metrics into Superset metrics.
"""

# pylint: disable=consider-using-f-string

import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set

import sqlglot
from sqlglot import Expression, ParseError, exp, parse_one
from sqlglot.expressions import Alias, Distinct, Identifier, Join, Select, Table, Where
from sqlglot.optimizer import traverse_scope

from preset_cli.api.clients.dbt import (
    FilterSchema,
    MFMetricWithSQLSchema,
    MFSQLEngine,
    ModelSchema,
    OGMetricSchema,
)
from preset_cli.api.clients.superset import SupersetMetricDefinition
from preset_cli.cli.superset.sync.dbt.exposures import ModelKey
from preset_cli.cli.superset.sync.dbt.lib import parse_metric_meta

_logger = logging.getLogger(__name__)

# dbt => sqlglot
DIALECT_MAP = {
    MFSQLEngine.BIGQUERY: "bigquery",
    MFSQLEngine.DUCKDB: "duckdb",
    MFSQLEngine.REDSHIFT: "redshift",
    MFSQLEngine.POSTGRES: "postgres",
    MFSQLEngine.SNOWFLAKE: "snowflake",
    MFSQLEngine.DATABRICKS: "databricks",
    MFSQLEngine.TRINO: "trino",
}


# pylint: disable=too-many-locals
def get_metric_expression(metric_name: str, metrics: Dict[str, OGMetricSchema]) -> str:
    """
    Return a SQL expression for a given dbt metric using sqlglot.
    """
    if metric_name not in metrics:
        raise Exception(f"Invalid metric {metric_name}")

    metric = metrics[metric_name]
    if "calculation_method" in metric:
        # dbt >= 1.3
        type_ = metric["calculation_method"]
        sql = metric["expression"]
    elif "sql" in metric:
        # dbt < 1.3
        type_ = metric["type"]
        sql = metric["sql"]
    else:
        raise Exception(f"Unable to generate metric expression from: {metric}")

    if metric.get("filters"):
        sql = apply_filters(sql, metric["filters"])

    simple_mappings = {
        "count": "COUNT",
        "sum": "SUM",
        "average": "AVG",
        "min": "MIN",
        "max": "MAX",
    }

    if type_ in simple_mappings:
        function = simple_mappings[type_]
        return f"{function}({sql})"

    if type_ == "count_distinct":
        return f"COUNT(DISTINCT {sql})"

    if type_ in {"expression", "derived"}:
        if metric.get("skip_parsing"):
            return sql.strip()

        try:
            expression = sqlglot.parse_one(sql, dialect=metric["dialect"])
            tokens = expression.find_all(exp.Column)

            for token in tokens:
                if token.sql() in metrics:
                    parent_sql = get_metric_expression(token.sql(), metrics)
                    parent_expression = sqlglot.parse_one(
                        parent_sql,
                        dialect=metric["dialect"],
                    )
                    token.replace(parent_expression)

            return expression.sql(dialect=metric["dialect"])
        except ParseError:
            sql = replace_metric_syntax(sql, metric["depends_on"], metrics)
            return sql

    sorted_metric = dict(sorted(metric.items()))
    raise Exception(f"Unable to generate metric expression from: {sorted_metric}")


def apply_filters(sql: str, filters: List[FilterSchema]) -> str:
    """
    Apply filters to SQL expression.
    """
    condition = " AND ".join(
        "{field} {operator} {value}".format(**filter_) for filter_ in filters
    )
    return f"CASE WHEN {condition} THEN {sql} END"


def is_derived(metric: OGMetricSchema) -> bool:
    """
    Return if the metric is derived.
    """
    return (
        metric.get("calculation_method") == "derived"  # dbt >= 1.3
        or metric.get("type") == "expression"  # dbt < 1.3
        or metric.get("type") == "derived"  # WTF dbt Cloud
    )


def get_metrics_for_model(
    model: ModelSchema,
    metrics: List[OGMetricSchema],
) -> List[OGMetricSchema]:
    """
    Given a list of metrics, return those that are based on a given model.
    """
    metric_map = {metric["unique_id"]: metric for metric in metrics}
    related_metrics = []

    for metric in metrics:
        parents = set()
        queue = [metric]
        while queue:
            node = queue.pop()
            depends_on = node["depends_on"]
            if is_derived(node):
                queue.extend(metric_map[parent] for parent in depends_on)
            else:
                parents.update(depends_on)

        if len(parents) > 1:
            _logger.warning(
                "Metric %s cannot be calculated because it depends on multiple models: %s",
                metric["name"],
                ", ".join(sorted(parents)),
            )
            continue

        if parents == {model["unique_id"]}:
            related_metrics.append(metric)

    return related_metrics


def get_metric_models(unique_id: str, metrics: List[OGMetricSchema]) -> Set[str]:
    """
    Given a metric, return the models it depends on.
    """
    metric_map = {metric["unique_id"]: metric for metric in metrics}
    metric = metric_map[unique_id]
    depends_on = metric["depends_on"]

    if is_derived(metric):
        return {
            model
            for parent in depends_on
            for model in get_metric_models(parent, metrics)
        }

    return set(depends_on)


def get_metric_definition(
    metric_name: str,
    metrics: List[OGMetricSchema],
) -> SupersetMetricDefinition:
    """
    Build a Superset metric definition from an OG (< 1.6) dbt metric.
    """
    metric_map = {metric["name"]: metric for metric in metrics}
    metric = metric_map[metric_name]
    metric_meta = parse_metric_meta(metric)
    final_metric_name = metric_meta["metric_name_override"] or metric_name

    return {
        "expression": get_metric_expression(metric_name, metric_map),
        "metric_name": final_metric_name,
        "metric_type": (metric.get("type") or metric.get("calculation_method")),
        "verbose_name": metric.get("label", final_metric_name),
        "description": metric.get("description", ""),
        "extra": json.dumps(metric_meta["meta"]),
        **metric_meta["kwargs"],  # type: ignore
    }


def get_superset_metrics_per_model(
    og_metrics: List[OGMetricSchema],
    sl_metrics: Optional[List[MFMetricWithSQLSchema]] = None,
) -> Dict[str, List[SupersetMetricDefinition]]:
    """
    Build a dictionary of Superset metrics for each dbt model.
    """
    superset_metrics = defaultdict(list)
    for metric in og_metrics:
        # dbt supports creating derived metrics with raw syntax. In case the metric doesn't
        # rely on other metrics (or rely on other metrics that aren't associated with any
        # model), it's required to specify the dataset the metric should be associated with
        # under the ``meta.superset.model`` key. If the derived metric is just an expression
        # with no dependency, it's not required to parse the metric SQL.
        if model := metric.get("meta", {}).get("superset", {}).pop("model", None):
            if len(metric["depends_on"]) == 0:
                metric["skip_parsing"] = True
        else:
            metric_models = get_metric_models(metric["unique_id"], og_metrics)
            if len(metric_models) == 0:
                _logger.warning(
                    "Metric %s cannot be calculated because it's not associated with any model."
                    " Please specify the model under metric.meta.superset.model.",
                    metric["name"],
                )
                continue

            if len(metric_models) != 1:
                _logger.warning(
                    "Metric %s cannot be calculated because it depends on multiple models: %s",
                    metric["name"],
                    ", ".join(sorted(metric_models)),
                )
                continue
            model = metric_models.pop()

        metric_definition = get_metric_definition(
            metric["name"],
            og_metrics,
        )
        superset_metrics[model].append(metric_definition)

    for sl_metric in sl_metrics or []:
        metric_definition = convert_metric_flow_to_superset(sl_metric)
        model = sl_metric["model"]
        superset_metrics[model].append(metric_definition)

    return superset_metrics


def extract_aliases(parsed_query: Expression) -> Dict[str, str]:
    """
    Extract column aliases from a SQL query.
    """
    aliases = {}
    for expression in parsed_query.find_all(Alias):
        alias_name = expression.alias
        expression_text = expression.this.sql()
        aliases[alias_name] = expression_text

    return aliases


def convert_query_to_projection(sql: str, dialect: MFSQLEngine) -> str:
    """
    Convert a MetricFlow compiled SQL to a projection.
    """
    # Parse the query using the given dialect
    parsed_query = parse_one(sql, dialect=DIALECT_MAP.get(dialect))

    # Extract aliases from the inner query if there's a subquery
    scopes = traverse_scope(parsed_query)
    has_subquery = len(scopes) > 1
    aliases = extract_aliases(scopes[0].expression) if has_subquery else {}

    # Locate the metric expression
    select_expression = parsed_query.find(Select)
    if select_expression.find(Join):
        raise ValueError("Unable to convert metrics with JOINs")

    # Ensure there's only one expression in the SELECT clause
    projection = select_expression.args.get("expressions", [])
    if len(projection) > 1:
        raise ValueError("Unable to convert metrics with multiple selected expressions")

    metric_expression = (
        projection[0].this if isinstance(projection[0], Alias) else projection[0]
    )

    # Find the WHERE clause and convert it to a CASE statement
    where_expression = parsed_query.find(Where)
    if where_expression:
        # Handle DISTINCT removal, if present
        if hasattr(metric_expression, "this") and isinstance(
            metric_expression.this,
            Expression,
        ):
            for node, _, _ in metric_expression.this.walk():
                if isinstance(node, Distinct) and node.expressions:
                    node.replace(node.expressions[0])
        else:
            _logger.warning(
                f"""Metric expression type {type(metric_expression.this)} is not iterable.
                Skipping DISTINCT check.""",
            )

        # Replace aliases in the WHERE clause with their original expressions
        for node, _, _ in where_expression.walk():
            if isinstance(node, Identifier) and node.sql() in aliases:
                node.replace(parse_one(aliases[node.sql()]))

    # Replace aliases in the metric expression with their original expressions
    for node, _, _ in metric_expression.walk():
        # Handle `NULLIF` replacement
        if node.sql().startswith("NULLIF("):
            if node.this:
                node.replace(node.this)

        # Replace identifiers with their alias definitions
        if isinstance(node, Identifier) and node.sql() in aliases:
            tree_alias = parse_one(aliases[node.sql()])
            case_expr = tree_alias.find(exp.Case)
            if case_expr and where_expression:
                case_condition = case_expr.args.get("ifs", [{}])[0].get("this")
                if case_condition:
                    full_condition = case_condition.and_(where_expression.this)
                    case_expr.args["ifs"][0]["this"] = full_condition
            node.replace(tree_alias)

    # Return the transformed SQL query as a string
    str_metric_expression = metric_expression.sql(dialect=DIALECT_MAP.get(dialect))
    str_metric_expression = str_metric_expression.replace("TODAY()", "today()")
    str_metric_expression = str_metric_expression.replace(
        "TOSTARTOFMONTH",
        "toStartOfMonth",
    )

    return str_metric_expression


def convert_metric_flow_to_superset(
    sl_metric: MFMetricWithSQLSchema,
) -> SupersetMetricDefinition:
    """
    Convert a MetricFlow metric to a Superset metric.

    Before MetricFlow we could build the metrics based on the metadata returned by the
    GraphQL API. With MetricFlow we only have access to the compiled SQL used to
    compute the metric, so we need to parse it and build a single projection for
    Superset.

    For example, this:

        SELECT
            SUM(order_count) AS large_order
        FROM (
            SELECT
                order_total AS order_id__order_total_dim
                , 1 AS order_count
            FROM `dbt-tutorial-347100`.`dbt_beto`.`orders` orders_src_106
        ) subq_796
        WHERE order_id__order_total_dim >= 20

    Becomes:

        SUM(CASE WHEN order_total > 20 THEN 1 END)

    """
    metric_meta = parse_metric_meta(sl_metric)
    return {
        "expression": convert_query_to_projection(
            sl_metric["sql"],
            sl_metric["dialect"],
        ),
        "metric_name": metric_meta["metric_name_override"] or sl_metric["name"],
        "metric_type": sl_metric["type"],
        "verbose_name": sl_metric["label"],
        "description": sl_metric["description"],
        "extra": json.dumps(metric_meta["meta"]),
        **metric_meta["kwargs"],  # type: ignore
    }


def get_models_from_sql(
    sql: str,
    dialect: MFSQLEngine,
    model_map: Dict[ModelKey, ModelSchema],
) -> Optional[List[ModelSchema]]:
    """
    Return the model associated with a SQL query.
    """
    parsed_query = parse_one(sql, dialect=DIALECT_MAP.get(dialect))
    sources = list(parsed_query.find_all(Table))

    for table in sources:
        if ModelKey(table.db, table.name) not in model_map:
            return None

    return [model_map[ModelKey(table.db, table.name)] for table in sources]


def replace_metric_syntax(
    sql: str,
    dependencies: List[str],
    metrics: Dict[str, OGMetricSchema],
) -> str:
    """
    Replace metric keys with their SQL syntax.
    This method is a fallback in case ``sqlglot`` raises a ``ParseError``.
    """
    for parent_metric in dependencies:
        parent_metric_name = parent_metric.split(".")[-1]
        pattern = r"\b" + re.escape(parent_metric_name) + r"\b"
        parent_metric_syntax = get_metric_expression(
            parent_metric_name,
            metrics,
        )
        sql = re.sub(pattern, parent_metric_syntax, sql)

    return sql.strip()
