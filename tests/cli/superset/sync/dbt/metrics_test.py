"""
Tests for metrics.
"""

# pylint: disable=line-too-long, too-many-lines

from typing import Dict

import pytest
from pytest_mock import MockerFixture

from preset_cli.api.clients.dbt import (
    MFMetricWithSQLSchema,
    MFSQLEngine,
    OGMetricSchema,
)
from preset_cli.cli.superset.sync.dbt.exposures import ModelKey
from preset_cli.cli.superset.sync.dbt.metrics import (
    convert_metric_flow_to_superset,
    get_metric_expression,
    get_metric_models,
    get_metrics_for_model,
    get_models_from_sql,
    get_superset_metrics_per_model,
)


def test_get_metric_expression() -> None:
    """
    Tests for ``get_metric_expression``.
    """
    metric_schema = OGMetricSchema()
    metrics: Dict[str, OGMetricSchema] = {
        "one": metric_schema.load(
            {
                "type": "count",
                "sql": "user_id",
                "filters": [
                    {"field": "is_paying", "operator": "is", "value": "true"},
                    {"field": "lifetime_value", "operator": ">=", "value": "100"},
                    {"field": "company_name", "operator": "!=", "value": "'Acme, Inc'"},
                    {"field": "signup_date", "operator": ">=", "value": "'2020-01-01'"},
                ],
                "dialect": "postgres",
            },
        ),
        "two": metric_schema.load(
            {
                "type": "count_distinct",
                "sql": "user_id",
                "dialect": "postgres",
            },
        ),
        "three": metric_schema.load(
            {
                "type": "expression",
                "sql": "one - two",
                "dialect": "postgres",
            },
        ),
        "four": metric_schema.load(
            {
                "type": "hllsketch",
                "sql": "user_id",
                "dialect": "postgres",
            },
        ),
        "load_fill_by_weight": metric_schema.load(
            {
                "depends_on": [
                    "metric.breakthrough_dw.load_weight_lbs",
                    "metric.breakthrough_dw.load_weight_capacity_lbs",
                ],
                "description": "The Load Fill by Weight",
                "filters": [],
                "label": "Load Fill by Weight",
                "meta": {},
                "name": "load_fill_by_weight",
                "sql": "load_weight_lbs / load_weight_capacity_lbs",
                "type": "derived",
                "unique_id": "metric.breakthrough_dw.load_fill_by_weight",
                "dialect": "postgres",
            },
        ),
    }
    assert get_metric_expression("one", metrics) == (
        "COUNT(CASE WHEN is_paying is true AND lifetime_value >= 100 AND "
        "company_name != 'Acme, Inc' AND signup_date >= '2020-01-01' THEN user_id END)"
    )

    assert get_metric_expression("two", metrics) == "COUNT(DISTINCT user_id)"

    assert get_metric_expression("three", metrics) == (
        "COUNT(CASE WHEN is_paying IS TRUE AND lifetime_value >= 100 AND "
        "company_name <> 'Acme, Inc' AND signup_date >= '2020-01-01' THEN user_id END) "
        "- COUNT(DISTINCT user_id)"
    )

    assert (
        get_metric_expression("load_fill_by_weight", metrics)
        == "load_weight_lbs / load_weight_capacity_lbs"
    )

    with pytest.raises(Exception) as excinfo:
        get_metric_expression("four", metrics)
    assert str(excinfo.value) == (
        "Unable to generate metric expression from: "
        "{'dialect': 'postgres', 'sql': 'user_id', 'type': 'hllsketch'}"
    )

    with pytest.raises(Exception) as excinfo:
        get_metric_expression("five", metrics)
    assert str(excinfo.value) == "Invalid metric five"

    with pytest.raises(Exception) as excinfo:
        get_metric_expression("six", {"six": {}})  # type: ignore
    assert str(excinfo.value) == "Unable to generate metric expression from: {}"


def test_get_metric_expression_new_schema() -> None:
    """
    Test ``get_metric_expression`` with the dbt 1.3 schema.

    See https://docs.getdbt.com/guides/migration/versions/upgrading-to-v1.3#for-users-of-dbt-metrics
    """
    metric_schema = OGMetricSchema()
    metrics: Dict[str, OGMetricSchema] = {
        "one": metric_schema.load(
            {
                "calculation_method": "count",
                "expression": "user_id",
                "filters": [
                    {"field": "is_paying", "operator": "is", "value": "true"},
                    {"field": "lifetime_value", "operator": ">=", "value": "100"},
                    {"field": "company_name", "operator": "!=", "value": "'Acme, Inc'"},
                    {"field": "signup_date", "operator": ">=", "value": "'2020-01-01'"},
                ],
            },
        ),
    }
    assert get_metric_expression("one", metrics) == (
        "COUNT(CASE WHEN is_paying is true AND lifetime_value >= 100 AND "
        "company_name != 'Acme, Inc' AND signup_date >= '2020-01-01' THEN user_id END)"
    )


def test_get_metric_expression_derived_legacy() -> None:
    """
    Test ``get_metric_expression`` with derived metrics created using a legacy dbt version.
    """
    metric_schema = OGMetricSchema()
    metrics: Dict[str, OGMetricSchema] = {
        "revenue_verbose_name_from_dbt": metric_schema.load(
            {
                "name": "revenue_verbose_name_from_dbt",
                "expression": "price_each",
                "description": "revenue.",
                "calculation_method": "sum",
                "unique_id": "metric.postgres.revenue_verbose_name_from_dbt",
                "label": "Sales Revenue Metric and this is the dbt label",
                "depends_on": ["model.postgres.vehicle_sales"],
                "metrics": [],
                "created_at": 1701101973.269536,
                "resource_type": "metric",
                "fqn": ["postgres", "revenue_verbose_name_from_dbt"],
                "model": "ref('vehicle_sales')",
                "path": "schema.yml",
                "package_name": "postgres",
                "original_file_path": "models/schema.yml",
                "refs": [{"name": "vehicle_sales", "package": None, "version": None}],
                "time_grains": [],
                "model_unique_id": None,
                "dialect": "postgres",
            },
        ),
        "derived_metric": metric_schema.load(
            {
                "name": "derived_metric",
                "expression": "revenue_verbose_name_from_dbt * 1.1",
                "description": "",
                "calculation_method": "derived",
                "unique_id": "metric.postgres.derived_metric",
                "label": "Dervied Metric",
                "depends_on": ["metric.postgres.revenue_verbose_name_from_dbt"],
                "metrics": [["revenue_verbose_name_from_dbt"]],
                "created_at": 1704299520.144628,
                "resource_type": "metric",
                "fqn": ["postgres", "derived_metric"],
                "model": None,
                "path": "schema.yml",
                "package_name": "bigquery",
                "original_file_path": "models/schema.yml",
                "refs": [],
                "time_grains": [],
                "model_unique_id": None,
                "config": {"enabled": True, "group": None},
                "dialect": "bigquery",
            },
        ),
        "another_derived_metric": metric_schema.load(
            {
                "name": "another_derived_metric",
                "expression": """
SAFE_DIVIDE(
        SUM(
          IF(
            `product_line` = "Classic Cars",
            price_each * 0.80,
            price_each * 0.70
          )
        ),
        revenue_verbose_name_from_dbt
      )
""",
                "description": "",
                "dialect": "bigquery",
                "calculation_method": "derived",
                "unique_id": "metric.postgres.another_derived_metric",
                "label": "Another Dervied Metric",
                "depends_on": ["metric.postgres.revenue_verbose_name_from_dbt"],
                "metrics": [["revenue_verbose_name_from_dbt"]],
                "created_at": 1704299520.144628,
                "resource_type": "metric",
                "fqn": ["postgres", "derived_metric"],
                "model": None,
                "path": "schema.yml",
                "package_name": "postgres",
                "original_file_path": "models/schema.yml",
                "refs": [],
                "time_grains": [],
                "model_unique_id": None,
                "config": {"enabled": True, "group": None},
            },
        ),
    }
    unique_id = "derived_metric"
    result = get_metric_expression(unique_id, metrics)
    assert result == "SUM(price_each) * 1.1"

    unique_id = "another_derived_metric"
    result = get_metric_expression(unique_id, metrics)
    assert (
        result
        == "SAFE_DIVIDE(SUM(IF(`product_line` = 'Classic Cars', price_each * 0.80, price_each * 0.70)), SUM(price_each))"
    )


def test_get_metrics_for_model(mocker: MockerFixture) -> None:
    """
    Test ``get_metrics_for_model``.
    """
    _logger = mocker.patch("preset_cli.cli.superset.sync.dbt.metrics._logger")

    metrics = [
        {
            "unique_id": "metric.superset.a",
            "depends_on": ["model.superset.table"],
            "name": "a",
        },
        {
            "unique_id": "metric.superset.b",
            "depends_on": ["model.superset.table"],
            "name": "b",
        },
        {
            "unique_id": "metric.superset.c",
            "depends_on": ["model.superset.other_table"],
            "name": "c",
        },
        {
            "unique_id": "metric.superset.d",
            "depends_on": ["metric.superset.a", "metric.superset.b"],
            "name": "d",
            "calculation_method": "derived",
        },
        {
            "unique_id": "metric.superset.e",
            "depends_on": ["metric.superset.a", "metric.superset.c"],
            "name": "e",
            "calculation_method": "derived",
        },
    ]

    model = {"unique_id": "model.superset.table"}
    assert get_metrics_for_model(model, metrics) == [  # type: ignore
        {
            "unique_id": "metric.superset.a",
            "depends_on": ["model.superset.table"],
            "name": "a",
        },
        {
            "unique_id": "metric.superset.b",
            "depends_on": ["model.superset.table"],
            "name": "b",
        },
        {
            "unique_id": "metric.superset.d",
            "depends_on": ["metric.superset.a", "metric.superset.b"],
            "name": "d",
            "calculation_method": "derived",
        },
    ]
    _logger.warning.assert_called_with(
        "Metric %s cannot be calculated because it depends on multiple models: %s",
        "e",
        "model.superset.other_table, model.superset.table",
    )

    model = {"unique_id": "model.superset.other_table"}
    assert get_metrics_for_model(model, metrics) == [  # type: ignore
        {
            "unique_id": "metric.superset.c",
            "depends_on": ["model.superset.other_table"],
            "name": "c",
        },
    ]


def test_get_metrics_derived_dbt_core() -> None:
    """
    Test derived metrics in dbt Core.
    """

    metrics = [
        {
            "name": "paying_customers",
            "resource_type": "metric",
            "package_name": "jaffle_shop",
            "path": "schema.yml",
            "original_file_path": "models/schema.yml",
            "unique_id": "metric.jaffle_shop.paying_customers",
            "fqn": ["jaffle_shop", "paying_customers"],
            "description": "",
            "label": "Customers who bought something",
            "calculation_method": "count",
            "expression": "customer_id",
            "filters": [{"field": "number_of_orders", "operator": ">", "value": "0"}],
            "time_grains": [],
            "dimensions": [],
            "timestamp": None,
            "window": None,
            "model": "ref('customers')",
            "model_unique_id": None,
            "meta": {},
            "tags": [],
            "config": {"enabled": True},
            "unrendered_config": {},
            "sources": [],
            "depends_on": ["model.jaffle_shop.customers"],
            "refs": [["customers"]],
            "metrics": [],
            "created_at": 1680229920.1190348,
        },
        {
            "name": "total_customers",
            "resource_type": "metric",
            "package_name": "jaffle_shop",
            "path": "schema.yml",
            "original_file_path": "models/schema.yml",
            "unique_id": "metric.jaffle_shop.total_customers",
            "fqn": ["jaffle_shop", "total_customers"],
            "description": "",
            "label": "Total customers",
            "calculation_method": "count",
            "expression": "customer_id",
            "filters": [],
            "time_grains": [],
            "dimensions": [],
            "timestamp": None,
            "window": None,
            "model": "ref('customers')",
            "model_unique_id": None,
            "meta": {},
            "tags": [],
            "config": {"enabled": True},
            "unrendered_config": {},
            "sources": [],
            "depends_on": ["model.jaffle_shop.customers"],
            "refs": [["customers"]],
            "metrics": [],
            "created_at": 1680229920.122923,
        },
        {
            "name": "ratio_of_paying_customers",
            "resource_type": "metric",
            "package_name": "jaffle_shop",
            "path": "schema.yml",
            "original_file_path": "models/schema.yml",
            "unique_id": "metric.jaffle_shop.ratio_of_paying_customers",
            "fqn": ["jaffle_shop", "ratio_of_paying_customers"],
            "description": "",
            "label": "Percentage of paying customers",
            "calculation_method": "derived",
            "expression": "paying_customers / total_customers",
            "filters": [],
            "time_grains": [],
            "dimensions": [],
            "timestamp": None,
            "window": None,
            "model": None,
            "model_unique_id": None,
            "meta": {},
            "tags": [],
            "config": {"enabled": True},
            "unrendered_config": {},
            "sources": [],
            "depends_on": [
                "metric.jaffle_shop.paying_customers",
                "metric.jaffle_shop.total_customers",
            ],
            "refs": [],
            "metrics": [["paying_customers"], ["total_customers"]],
            "created_at": 1680230520.212716,
        },
    ]
    model = {"unique_id": "model.jaffle_shop.customers"}
    assert get_metrics_for_model(model, metrics) == metrics  # type: ignore


def test_get_metrics_derived_dbt_cloud() -> None:
    """
    Test derived metrics in dbt Cloud.
    """
    metrics = [
        {
            "depends_on": ["model.jaffle_shop.customers"],
            "description": "The number of paid customers using the product",
            "filters": [{"field": "number_of_orders", "operator": "=", "value": "0"}],
            "label": "New customers",
            "meta": {},
            "name": "new_customers",
            "sql": "customer_id",
            "type": "count",
            "unique_id": "metric.jaffle_shop.new_customers",
        },
        {
            "depends_on": ["model.jaffle_shop.customers"],
            "description": "",
            "filters": [{"field": "number_of_orders", "operator": ">", "value": "0"}],
            "label": "Customers who bought something",
            "meta": {},
            "name": "paying_customers",
            "sql": "customer_id",
            "type": "count",
            "unique_id": "metric.jaffle_shop.paying_customers",
        },
        {
            "depends_on": [
                "metric.jaffle_shop.paying_customers",
                "metric.jaffle_shop.total_customers",
            ],
            "description": "",
            "filters": [],
            "label": "Percentage of paying customers",
            "meta": {},
            "name": "ratio_of_paying_customers",
            "sql": "paying_customers / total_customers",
            "type": "derived",
            "unique_id": "metric.jaffle_shop.ratio_of_paying_customers",
        },
        {
            "depends_on": ["model.jaffle_shop.customers"],
            "description": "",
            "filters": [],
            "label": "Total customers",
            "meta": {},
            "name": "total_customers",
            "sql": "customer_id",
            "type": "count",
            "unique_id": "metric.jaffle_shop.total_customers",
        },
    ]
    model = {"unique_id": "model.jaffle_shop.customers"}
    assert get_metrics_for_model(model, metrics) == metrics  # type: ignore


def test_get_metric_models() -> None:
    """
    Tests for ``get_metric_models``.
    """
    metric_schema = OGMetricSchema()
    metrics = [
        metric_schema.load(
            {
                "unique_id": "metric.superset.a",
                "depends_on": ["model.superset.table"],
                "name": "a",
            },
        ),
        metric_schema.load(
            {
                "unique_id": "metric.superset.b",
                "depends_on": ["model.superset.table"],
                "name": "b",
            },
        ),
        metric_schema.load(
            {
                "unique_id": "metric.superset.c",
                "depends_on": ["model.superset.other_table"],
                "name": "c",
            },
        ),
        metric_schema.load(
            {
                "unique_id": "metric.superset.d",
                "depends_on": ["metric.superset.a", "metric.superset.b"],
                "name": "d",
                "calculation_method": "derived",
            },
        ),
        metric_schema.load(
            {
                "unique_id": "metric.superset.e",
                "depends_on": ["metric.superset.a", "metric.superset.c"],
                "name": "e",
                "calculation_method": "derived",
            },
        ),
    ]
    assert get_metric_models("metric.superset.a", metrics) == {"model.superset.table"}
    assert get_metric_models("metric.superset.b", metrics) == {"model.superset.table"}
    assert get_metric_models("metric.superset.c", metrics) == {
        "model.superset.other_table",
    }
    assert get_metric_models("metric.superset.d", metrics) == {"model.superset.table"}
    assert get_metric_models("metric.superset.e", metrics) == {
        "model.superset.other_table",
        "model.superset.table",
    }


def test_convert_metric_flow_to_superset(mocker: MockerFixture) -> None:
    """
    Test the ``convert_metric_flow_to_superset`` function.
    """
    mocker.patch(
        "preset_cli.cli.superset.sync.dbt.metrics.convert_query_to_projection",
        side_effect=["SUM(order_total)", "SUM(price_each)"],
    )
    mf_metric_schema = MFMetricWithSQLSchema()
    semantic_metric = mf_metric_schema.load(
        {
            "name": "sales",
            "description": "All sales",
            "label": "Sales",
            "type": "SIMPLE",
            "sql": "SELECT SUM(order_total) AS order_total FROM orders",
            "dialect": MFSQLEngine.BIGQUERY,
            "meta": {
                "superset": {
                    "d3format": "0.2f",
                },
            },
        },
    )

    assert convert_metric_flow_to_superset(semantic_metric) == {
        "expression": "SUM(order_total)",
        "metric_name": "sales",
        "metric_type": "SIMPLE",
        "verbose_name": "Sales",
        "description": "All sales",
        "d3format": "0.2f",
        "extra": "{}",
    }

    # Metric key override
    other_semantic_metric = mf_metric_schema.load(
        {
            "name": "revenue",
            "description": "Total revenue in the period",
            "label": "Total Revenue",
            "type": "SIMPLE",
            "sql": "SELECT SUM(price_each) AS price_each FROM orders",
            "dialect": MFSQLEngine.BIGQUERY,
            "meta": {
                "superset": {
                    "metric_name": "preset_specific_key",
                },
            },
        },
    )

    assert convert_metric_flow_to_superset(other_semantic_metric) == {
        "expression": "SUM(price_each)",
        "metric_name": "preset_specific_key",
        "metric_type": "SIMPLE",
        "verbose_name": "Total Revenue",
        "description": "Total revenue in the period",
        "extra": "{}",
    }


def test_get_models_from_sql() -> None:
    """
    Test the ``get_models_from_sql`` function.
    """
    model_map = {
        ModelKey("schema", "table"): {"name": "table"},
        ModelKey("schema", "a"): {"name": "a"},
        ModelKey("schema", "b"): {"name": "b"},
    }

    assert get_models_from_sql(
        "SELECT 1 FROM project.schema.table",
        MFSQLEngine.BIGQUERY,
        model_map,  # type: ignore
    ) == [{"name": "table"}]

    assert get_models_from_sql(
        "SELECT 1 FROM schema.a JOIN schema.b",
        MFSQLEngine.BIGQUERY,
        model_map,  # type: ignore
    ) == [{"name": "a"}, {"name": "b"}]

    assert (
        get_models_from_sql("SELECT 1 FROM schema.c", MFSQLEngine.BIGQUERY, {}) is None
    )


def test_get_superset_metrics_per_model() -> None:
    """
    Tests for the ``get_superset_metrics_per_model`` function.
    """
    mf_metric_schema = MFMetricWithSQLSchema()
    og_metric_schema = OGMetricSchema()

    og_metrics = [
        og_metric_schema.load(obj)
        for obj in [
            {
                "name": "sales",
                "unique_id": "sales",
                "depends_on": ["orders"],
                "calculation_method": "sum",
                "expression": "1",
                "label": "Sales",
                "meta": {},
            },
            {
                "name": "multi-model",
                "unique_id": "multi-model",
                "depends_on": ["a", "b"],
                "calculation_method": "derived",
                "meta": {},
            },
            {
                "name": "a",
                "unique_id": "a",
                "depends_on": ["orders"],
                "calculation_method": "sum",
                "expression": "1",
                "meta": {
                    "superset": {
                        "warning_text": "caution",
                    },
                },
            },
            {
                "name": "b",
                "unique_id": "b",
                "depends_on": ["customers"],
                "calculation_method": "sum",
                "expression": "1",
                "meta": {
                    "superset": {
                        "warning_text": "meta under config",
                    },
                },
            },
            {
                "name": "to_be_updated",
                "label": "Preset Label",
                "unique_id": "to_be_updated",
                "depends_on": ["customers"],
                "calculation_method": "max",
                "expression": "1",
                "meta": {
                    "superset": {
                        "metric_name": "new_key",
                    },
                },
            },
        ]
    ]

    sl_metrics = [
        mf_metric_schema.load(obj)
        for obj in [
            {
                "name": "new",
                "description": "New metric",
                "label": "New Label",
                "type": "SIMPLE",
                "sql": "SELECT COUNT(1) FROM a.b.c",
                "dialect": MFSQLEngine.BIGQUERY,
                "model": "new-model",
                "meta": {},
            },
            {
                "name": "other_new",
                "description": "This is a test replacing the metric key",
                "label": "top Label",
                "type": "SIMPLE",
                "sql": "SELECT COUNT(1) FROM a.b.c",
                "dialect": MFSQLEngine.BIGQUERY,
                "model": "new-model",
                "meta": {
                    "superset": {
                        "metric_name": "preset_sl_key",
                    },
                },
            },
        ]
    ]

    assert get_superset_metrics_per_model(og_metrics, sl_metrics) == {
        "orders": [
            {
                "expression": "SUM(1)",
                "metric_name": "sales",
                "metric_type": "sum",
                "verbose_name": "Sales",
                "description": "",
                "extra": "{}",
            },
            {
                "expression": "SUM(1)",
                "metric_name": "a",
                "metric_type": "sum",
                "verbose_name": "a",
                "description": "",
                "warning_text": "caution",
                "extra": "{}",
            },
        ],
        "customers": [
            {
                "expression": "SUM(1)",
                "metric_name": "b",
                "metric_type": "sum",
                "verbose_name": "b",
                "description": "",
                "warning_text": "meta under config",
                "extra": "{}",
            },
            {
                "expression": "MAX(1)",
                "metric_name": "new_key",
                "metric_type": "max",
                "verbose_name": "Preset Label",
                "description": "",
                "extra": "{}",
            },
        ],
        "new-model": [
            {
                "expression": "COUNT(1)",
                "metric_name": "new",
                "metric_type": "SIMPLE",
                "verbose_name": "New Label",
                "description": "New metric",
                "extra": "{}",
            },
            {
                "expression": "COUNT(1)",
                "metric_name": "preset_sl_key",
                "metric_type": "SIMPLE",
                "verbose_name": "top Label",
                "description": "This is a test replacing the metric key",
                "extra": "{}",
            },
        ],
    }


def test_get_superset_metrics_per_model_og_derived(
    caplog: pytest.CaptureFixture[str],
) -> None:
    """
    Tests for the ``get_superset_metrics_per_model`` function
    with derived OG metrics.
    """
    og_metric_schema = OGMetricSchema()

    og_metrics = [
        og_metric_schema.load(
            {
                "name": "sales",
                "unique_id": "sales",
                "depends_on": ["orders"],
                "calculation_method": "sum",
                "expression": "1",
                "meta": {},
            },
        ),
        og_metric_schema.load(
            {
                "name": "revenue",
                "unique_id": "revenue",
                "depends_on": ["orders"],
                "calculation_method": "sum",
                "expression": "price_each",
                "meta": {},
            },
        ),
        og_metric_schema.load(
            {
                "name": "derived_metric_missing_model_info",
                "unique_id": "derived_metric_missing_model_info",
                "depends_on": [],
                "calculation_method": "derived",
                "expression": "price_each * 1.2",
                "meta": {},
            },
        ),
        og_metric_schema.load(
            {
                "name": "derived_metric_model_from_meta",
                "unique_id": "derived_metric_model_from_meta",
                "depends_on": [],
                "calculation_method": "derived",
                "expression": "(SUM(price_each)) * 1.2",
                "meta": {"superset": {"model": "customers"}},
            },
        ),
        og_metric_schema.load(
            {
                "name": "derived_metric_with_jinja",
                "unique_id": "derived_metric_with_jinja",
                "depends_on": [],
                "calculation_method": "derived",
                "expression": """
SUM(
    {% for x in filter_values('x_values') %}
        {{ + x_values }}
    {% endfor %}
)
""",
                "meta": {"superset": {"model": "customers"}},
            },
        ),
        og_metric_schema.load(
            {
                "name": "derived_metric_with_jinja_and_other_metric",
                "unique_id": "derived_metric_with_jinja_and_other_metric",
                "depends_on": ["sales"],
                "dialect": "postgres",
                "calculation_method": "derived",
                "expression": """
SUM(
    {% for x in filter_values('x_values') %}
        {{ my_sales + sales }}
    {% endfor %}
)
""",
                "meta": {},
            },
        ),
        og_metric_schema.load(
            {
                "name": "derived_combining_other_derived_including_jinja",
                "unique_id": "derived_combining_other_derived_including_jinja",
                "depends_on": ["derived_metric_with_jinja_and_other_metric", "revenue"],
                "dialect": "postgres",
                "calculation_method": "derived",
                "expression": "derived_metric_with_jinja_and_other_metric / revenue",
                "meta": {},
            },
        ),
        og_metric_schema.load(
            {
                "name": "simple_derived",
                "unique_id": "simple_derived",
                "depends_on": [],
                "dialect": "postgres",
                "calculation_method": "derived",
                "expression": "max(order_date)",
                "meta": {"superset": {"model": "customers"}},
            },
        ),
        og_metric_schema.load(
            {
                "name": "last_derived_example",
                "unique_id": "last_derived_example",
                "depends_on": ["simple_derived"],
                "dialect": "postgres",
                "calculation_method": "derived",
                "expression": "simple_derived - 1",
                "meta": {"superset": {"model": "customers"}},
            },
        ),
    ]

    result = get_superset_metrics_per_model(og_metrics, [])
    output_content = caplog.text
    assert (
        "Metric derived_metric_missing_model_info cannot be calculated because it's not associated with any model"
        in output_content
    )

    assert result == {
        "customers": [
            {
                "expression": "(SUM(price_each)) * 1.2",
                "metric_name": "derived_metric_model_from_meta",
                "metric_type": "derived",
                "verbose_name": "derived_metric_model_from_meta",
                "description": "",
                "extra": "{}",
            },
            {
                "expression": """SUM(
    {% for x in filter_values('x_values') %}
        {{ + x_values }}
    {% endfor %}
)""",
                "metric_name": "derived_metric_with_jinja",
                "metric_type": "derived",
                "verbose_name": "derived_metric_with_jinja",
                "description": "",
                "extra": "{}",
            },
            {
                "expression": "max(order_date)",
                "metric_name": "simple_derived",
                "metric_type": "derived",
                "verbose_name": "simple_derived",
                "description": "",
                "extra": "{}",
            },
            {
                "expression": "MAX(order_date) - 1",
                "metric_name": "last_derived_example",
                "metric_type": "derived",
                "verbose_name": "last_derived_example",
                "description": "",
                "extra": "{}",
            },
        ],
        "orders": [
            {
                "description": "",
                "expression": "SUM(1)",
                "extra": "{}",
                "metric_name": "sales",
                "metric_type": "sum",
                "verbose_name": "sales",
            },
            {
                "description": "",
                "expression": "SUM(price_each)",
                "extra": "{}",
                "metric_name": "revenue",
                "metric_type": "sum",
                "verbose_name": "revenue",
            },
            {
                "expression": """SUM(
    {% for x in filter_values('x_values') %}
        {{ my_sales + SUM(1) }}
    {% endfor %}
)""",
                "metric_name": "derived_metric_with_jinja_and_other_metric",
                "metric_type": "derived",
                "verbose_name": "derived_metric_with_jinja_and_other_metric",
                "description": "",
                "extra": "{}",
            },
            {
                "expression": """SUM(
    {% for x in filter_values('x_values') %}
        {{ my_sales + SUM(1) }}
    {% endfor %}
) / SUM(price_each)""",
                "metric_name": "derived_combining_other_derived_including_jinja",
                "metric_type": "derived",
                "verbose_name": "derived_combining_other_derived_including_jinja",
                "description": "",
                "extra": "{}",
            },
        ],
    }
