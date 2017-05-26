import xlrd
import datetime
from collections import defaultdict
import json
import os

formatted_trace_path = '/vagrant/test_data'
def prepare_trace_date(excel_path,
                        sheet_name,
                        project_name,
                        account,
                        start_date_col=1):
    wb = xlrd.open_workbook(excel_path)

    sh = wb.sheet_by_name(sheet_name)
    result = {}
    for rownum in xrange(sh.nrows):
        if rownum == 0:
            continue
        rowvalues = sh.row_values(rownum)
        account_data = rowvalues[0]
        if account_data != account:
            continue


        col_date = datetime.datetime(*xlrd.xldate_as_tuple(rowvalues[start_date_col], wb.datemode))
        start = col_date.strftime("%Y-%m-%d %H:%M:%S+00:00")
        value = float(rowvalues[3])
        units = ''

        single_record = {
            'start' : start,
            'value' : value,
            'estimated' : False
        }

        records = result.get('records', None)
        if not records:
            records = []
            result['records'] = records

        records.append(single_record)

    result['trace_id'] = project_name + "-" + account
    return result

excel_path = '/vagrant/test_data/Bright Power sample MnV data.xlsx'

trace_records_1 = prepare_trace_date(excel_path, 'Property I', 'Property I', 'Electric I')
trace_file = open(os.path.join(formatted_trace_path, "property1-trace.json"), 'wb')
trace_file.write(json.dumps(trace_records_1))

trace_records_2 = prepare_trace_date(excel_path, 'Property I', 'Property I', 'Gas I')
trace_file = open(os.path.join(formatted_trace_path, "property1-trace2.json"), 'wb')
trace_file.write(json.dumps(trace_records_1))

trace_records_3 = prepare_trace_date(excel_path, 'Property II', 'Property II', 'Electric I')
trace_file = open(os.path.join(formatted_trace_path, "property2-trace1.json"), 'wb')
trace_file.write(json.dumps(trace_records_3))

trace_records_4 = prepare_trace_date(excel_path, 'Property II', 'Property II', 'Electric II')
trace_file = open(os.path.join(formatted_trace_path, "property2-trace2.json"), 'wb')
trace_file.write(json.dumps(trace_records_4))

trace_records_5= prepare_trace_date(excel_path, 'Property III', 'Property III', 'Oil I')
trace_file = open(os.path.join(formatted_trace_path, "property3-trace1.json"), 'wb')
trace_file.write(json.dumps(trace_records_5))

trace_records_6 = prepare_trace_date(excel_path, 'Property III', 'Property III', 'Gas I')
trace_file = open(os.path.join(formatted_trace_path, "property3-trace2.json"), 'wb')
trace_file.write(json.dumps(trace_records_6))

trace_records_7 = prepare_trace_date(excel_path, 'Property IV', 'Property IV', 'Electric I')
trace_file = open(os.path.join(formatted_trace_path, "property4-trace1.json"), 'wb')
trace_file.write(json.dumps(trace_records_7))

trace_records_8 = prepare_trace_date(excel_path, 'Property IV', 'Property IV', 'Electric II')
trace_file = open(os.path.join(formatted_trace_path, "property4-trace2.json"), 'wb')
trace_file.write(json.dumps(trace_records_8))

trace_records_9 = prepare_trace_date(excel_path, 'Property IV', 'Property IV', 'Electric III')
trace_file = open(os.path.join(formatted_trace_path, "property4-trace2.json"), 'wb')
trace_file.write(json.dumps(trace_records_9))

trace_records_10 = prepare_trace_date(excel_path, 'Property IV', 'Property IV', 'Electric IV')
trace_file = open(os.path.join(formatted_trace_path, "property4-trace3.json"), 'wb')
trace_file.write(json.dumps(trace_records_10))

trace_records_11 = prepare_trace_date(excel_path, 'Property IV', 'Property IV', 'Gas and Oil I')
trace_file = open(os.path.join(formatted_trace_path, "property4-trace4.json"), 'wb')
trace_file.write(json.dumps(trace_records_11))

trace_records_12 = prepare_trace_date(excel_path, 'Property V', 'Property V', 'Electric I')
trace_file = open(os.path.join(formatted_trace_path, "property5-trace1.json"), 'wb')
trace_file.write(json.dumps(trace_records_12))

trace_records_13 = prepare_trace_date(excel_path, 'Property V', 'Property V', 'Gas I')
trace_file = open(os.path.join(formatted_trace_path, "property5-trace2.json"), 'wb')
trace_file.write(json.dumps(trace_records_13))

