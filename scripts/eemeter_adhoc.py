from collections import OrderedDict
import json
import pytz
import pandas as pd

from datetime import datetime
from eemeter.structures import EnergyTrace
from eemeter.io.serializers import ArbitraryStartSerializer
from eemeter.ee.meter import EnergyEfficiencyMeter
from eemeter.modeling.models import CaltrackMonthlyModel
from eemeter.modeling.formatters import ModelDataFormatter

def serialize_meter_input(project_id, trace, zipcode, retrofit_start_date, retrofit_end_date):

    data = OrderedDict([
        ("type", "SINGLE_TRACE_SIMPLE_PROJECT"),
        ("trace", trace_serializer(trace)),
        ("project", project_serializer(project_id, zipcode, retrofit_start_date, retrofit_end_date)),
    ])

    return data

def trace_serializer(trace):
    data = OrderedDict([
        ("type", "ARBITRARY_START"),
        ("interpretation", trace.interpretation),
        ("unit", trace.unit),
        ("trace_id", trace.trace_id),
        ("interval", trace.interval),
        ("records", [
            OrderedDict([
                ("start", start.isoformat()),
                ("value", record.value if pd.notnull(record.value) else None),
                ("estimated", bool(record.estimated)),
            ])
            for start, record in trace.data.iterrows()
        ]),
    ])
    return data

def project_serializer(project_id, zipcode, retrofit_start_date, retrofit_end_date):
    data = OrderedDict([
        ("type", "PROJECT_WITH_SINGLE_MODELING_PERIOD_GROUP"),
        ("zipcode", zipcode),
        ("project_id", project_id),
        ("modeling_period_group", OrderedDict([
            ("baseline_period", OrderedDict([
                ("start", None),
                ("end", retrofit_start_date.isoformat()),
            ])),
            ("reporting_period", OrderedDict([
                ("start", retrofit_end_date.isoformat()),
                ("end", None),
            ]))
        ]))
    ])
    return data

def parse_trace_json_data(trace_file):
    data = json.load(open(trace_file))
    records = data['records']
    trace_id = data['trace_id']

    new_records = []
    for rec in records:
        start = pytz.UTC.localize(datetime.strptime(rec['start'], "%Y-%m-%d %H:%M:%S+00:00"))
        rec['start'] = start
        new_records.append(rec)

    return (new_records, trace_id)

def run_meter(trace_file,
              project_id,
              zipcode,
              interpretation,
              unit,
              retrofit_start_date,
              retrofit_end_date,
              interval='daily'):
    """

    Parameters
    ----------
    trace_file
    project_id
    interpretation
    unit
    retrofit_start_date
    retrofit_end_date

    Returns
    -------

    """
    records, trace_id = parse_trace_json_data(trace_file)
    energy_trace = EnergyTrace(records=records, unit=unit,
                               interpretation=interpretation,
                               serializer=ArbitraryStartSerializer(),
                                trace_id=trace_id,
                                interval=interval)

    meter_input = serialize_meter_input(project_id,
                                        energy_trace,
                                        zipcode,
                                        retrofit_start_date,
                                        retrofit_end_date)

    meter = EnergyEfficiencyMeter()
    model = (CaltrackMonthlyModel, {"fit_cdd": False, "grid_search": True})
    formatter = (ModelDataFormatter, {"freq_str": "D"})

    meter_output = meter.evaluate(meter_input, model=model, formatter=formatter)


    print(json.dumps([(d['series'], d['description'])
                      for d in sorted(meter_output["derivatives"],
                                      key=lambda o: o['series'])], indent=2))


    with open('/vagrant/test_data/meter_output_example.json', 'w') as f:  # change this path if desired.
        json.dump(meter_output, f, indent=2)

run_meter('/vagrant/test_data/property5-trace1.json',
          'Property V',
          '10036',
          'ELECTRICITY_CONSUMPTION_SUPPLIED',
          'KWH',
          pytz.UTC.localize(datetime.strptime('2013-12-01 00:00:00+00:00', "%Y-%m-%d %H:%M:%S+00:00")),
          pytz.UTC.localize(datetime.strptime('2015-12-15 00:00:00+00:00', "%Y-%m-%d %H:%M:%S+00:00")),
          interval="billing"
          )