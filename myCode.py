"""
basic operations
"""
# set
{}, {}.add()
# string replace
my_str.replace("'", "\'")
# dict keys
for key in my_dict.keys():

def split_list_quote(set_):
    ct = 0
    for item in set_:
        ct += 1
        if ct == 1:
            sql_out = "'" + str(item) + "'"
        if ct > 1:
            sql_out += ', ' + "'" + str(item) + "'"

    return sql_out

"{0:.0%}".format(pid_i / pid_total)


-------------------------------------------
"""
numpy
"""
np.bincount(y)  Labels counts in y: [50 50 50]


-------------------------------------------
"""
dataframe 
"""
# sort_values
df_condition.sort_values(by=['excitation_id', 'condition_code'], inplace=True)
# string split -> list
list(x for x in df_excitation[df_excitation['id'] == excitation]['dot_id'])[0].split(',')
# reset index
df.reset_index()
# merge
dots_cond = pd.merge(dots_temp, dots_temp2, on='dot_id', how='inner')
df_merge = pd.merge(df_03006_gp, df_03006_close_gp, left_index=True, right_index=True, how='left')
# concat
orders_df = pd.concat([orders_df, orders_temp_df], sort=True)
# fillna
df.fillna(value={'prod_type': 666}, inplace=True)
# change column data type
comm_detail_df = comm_detail_df.astype({"prod_type": int})
# isin / update value in df / index matches in the example
df.loc[df[df['dot_id'].isin(superior_map['dot_id'])].index, 'c_type'] = df['c_type_super']
# drop columns
df.drop(columns=['c_type_infer', 'superior_dot_id', 'c_type_super'], inplace=True)
# apply / lambda
df['开始时间'] = df['开始时间'].apply(lambda x: get_first_day(pd.to_datetime(x)))
comm_df['base_actual'] = comm_df['base_actual'].apply(lambda x: x if x > 0 else 0)
# check if null 
pd.isnull(row['orders_detail_no'])
# rename
df.rename(columns={'开始时间': 'bonus_start','结束时间': 'bonus_end'}, inplace=True)
# map column values 
df_2['order_type'] = df['order_type'].map({'长短租': 1, '日租': 2, '整租': 0})
# drop na 
page_df.dropna(subset=['km'], inplace=True)
# tolist
value_list = df_2.values.tolist()
# series, np.where
partner_bonus_df['orders_no_combined'] = pd.Series(
            np.where(pd.isnull(partner_bonus_df['ordDetailId']), partner_bonus_df['reOrderId'],
                     partner_bonus_df['ordDetailId']))
# group by 
grouped = excitation_df.groupby('orders_no_combined')
df_03006_gp = pd.DataFrame(df_03006.groupby('客户编码')['本月分摊金额'].sum())
exc_df = pd.DataFrame(grouped['where_value'].agg(np.sum)).reset_index()
grouped = df.groupby([gp_by])
df_result = pd.DataFrame(columns=df.columns)
for name, group in grouped:
    gp_temp = group.sort_values(by=sort_by, ascending=acd)
    gp = gp_temp.head(1)
    df_result = df_result.append(gp)

pid_groupgy = df_pid_transfer.groupby('pid')
df_need_fix = pd.DataFrame()
for name, group in pid_groupgy:
    s = group['begin_time_ood'].value_counts()
    if len(s[s >= 2].index) == 0:
        continue
    else:
        print(name)
        selected_rows = group[group['begin_time_ood'].isin(s[s >= 2].index)].copy()
        df_need_fix = df_need_fix.append(selected_rows, ignore_index=True, sort=False)

# insert one column
exc_df.insert(loc=len(exc_df.columns), column='orders_no_combined', value=np.nan)
# append
df_final = df1.append(df2, sort=True)
# pd concat
exsiting_detail_df = pd.concat([daily_df, monthly_df], axis=0, ignore_index=True, sort=True)
# basics 
data['item'].count()
data['duration'].max()
data['duration'][data['item'] == 'call'].sum()
data['month'].value_counts()
data['network'].nunique()
count	Number of non-null observations
sum	Sum of values
mean	Mean of values
mad	Mean absolute deviation
median	Arithmetic median of values
min	Minimum
max	Maximum
mode	Mode
abs	Absolute Value
prod	Product of values
std	Unbiased standard deviation
var	Unbiased variance
sem	Unbiased standard error of the mean
skew	Unbiased skewness (3rd moment)
kurt	Unbiased kurtosis (4th moment)
quantile	Sample quantile (value at %)
cumsum	Cumulative sum
cumprod	Cumulative product
cummax	Cumulative maximum
cummin	Cumulative minimum

df['day_this'] = df[['day_this', 'order_days']].min(axis=1)

# drop duplicates
df_pid_bt.drop_duplicates(inplace=True)

-------------------------------------------
"""
interact with sql 
"""
def replace_table(table, engine, df, db, method='replace'):
    time_start = time.time()

    sql_read_table = "SELECT table_name FROM information_schema.tables WHERE table_schema = %s" % db
    tables = pd.read_sql_query(sql_read_table, engine)
    table_name = tables['table_name'].values
    # check if the table exists
    if table in table_name:
        print('table exists and will be replaced or updated')
    else:
        print('This is a new table and will be created')

    if method == 'replace':
        try:
            sql_clear_table = "truncate table {}".format(table)
            pd.read_sql(sql_clear_table, engine)
        except Exception as e:
            pass
        df.to_sql(table, engine, index=False, if_exists="append")
    elif method == 'append':
        df.to_sql(table, engine, index=False, if_exists="append")
    else:
        print("unrecognized manage table method, neither replace nor append")

    print('Write to Mysql table successfully!')
    time_end = time.time()
    print('totally cost', time_end - time_start)

    return None

def replace_table(table, engine, df, method='replace'):
    if method == 'replace':
        try:
            sql_clear_table = "truncate table {}".format(table)
            pd.read_sql(sql_clear_table, engine)
        except:
            pass
        df.to_sql(table, engine, index=False, if_exists="append")
    elif method == 'append':
        df.to_sql(table, engine, index=False, if_exists="append")
    else:
        print("unrecognized manage table method, neither replace nor append")

    print('Write to Mysql table successfully!')

    return None

table = 'excitation_table'
engine_168 = create_engine(engine_168_url)
data = job_excitation()
db = "'commissions'"

replace_table(table, engine_168, data, db)

-------------------------------------------
"""
datetime
"""
# datatime -> string strftime
df_excitation[df_excitation['id'] == excitation]['effect_start_time'].values[0].strftime('%Y-%m-%d')
# string -> datetime strptime -> date
datetime.strptime(exc_dict['effect_end_time'], '%Y-%m-%d').date()
# timestamp to datetime 
orders_df['begin_time'] = pd.to_datetime(orders_df['begin_time'])
# create timestamp
pd.Timestamp('1900-01-01T12')
# days between two date
days_passed = (date.today() - begin_date).days
# epoch time to datetime 
dt_object = datetime.fromtimestamp(timestamp)
# datetime to epoch time
dt.timestamp()
# date to datetime
datetime.combine(date.today(), datetime.min.time()) + timedelta(-30)

def start_end_date(begin_dt_s, end_dt_s):
    begin_dt = datetime(int(begin_dt_s[0:4]), int(begin_dt_s[4:6]), int(begin_dt_s[6:8]), 0, 0, 0)
    end_dt = datetime(int(end_dt_s[0:4]), int(end_dt_s[4:6]), int(end_dt_s[6:8]), 0, 0, 0)
    start = max(datetime.combine(begin_dt, datetime.min.time()).timestamp(), datetime(2019, 11, 1, 0, 0, 0).timestamp())
    end = min(datetime.combine(end_dt, datetime.min.time()).timestamp(), datetime(2019, 11, 23, 23, 59, 59).timestamp())

    return datetime.fromtimestamp(start).date(), datetime.fromtimestamp(end).date()


def get_first_day(dt, d_years=0, d_months=0):
    # 取当月第一天
    # d_years, d_months are "deltas" to apply to dt
    y, m = int(dt.year) + int(d_years), int(dt.month) + int(d_months)
    a, m = divmod(m - 1, 12)
    return date(y + a, m + 1, 1)


def get_last_day(dt):
    # 取当月最后一天
    return get_first_day(dt, 0, 1) + timedelta(-1)



-------------------------------------------
"""
shcedule
"""
if __name__ == '__main__':
    excitation_table_put()
    commision_detail_replace()
    first_day_job()

    # 每天下午18:30跑至前一天的激励订单
    schedule.every().day.at("18:30").do(excitation_table_put)
    # 每天下午19:30跑当前一天的佣金明细
    schedule.every().day.at("19:30").do(commision_detail_replace)
    # 每个月第一天的20:00将当月表存为历史表
    # 每个月第一天的20:00將base_comm_map存一次
    schedule.every().day.at("20:00").do(first_day_job)
    # 手动将上月的历史表中的数据从未发改为已发 / 或設置為某一天如15號


    while True:
        schedule.run_pending()
    pass


-------------------------------------------
"""
logging
"""
import logging

# logging 设置 操作日志 设置
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("process_maintenance.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


-------------------------------------------
""""
multi process
"""
import multiprocessing as mp

def split_df(df, n_part):
    df_len = df.shape[0]
    part_len = math.ceil(df_len / n_part)
    df_list = []
    for i in range(n_part):
        if i == n_part - 1:
            iloc_s = i*part_len
            iloc_e = df_len+1
        else:
            iloc_s = i*part_len
            iloc_e = i*part_len + part_len
        df_part = df.iloc[iloc_s:iloc_e]
        df_list.append(df_part)

    return df_list

result_list = []

def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)


def multi_p_task(df, n_part):
    print("multi processing begins")
    begin_time = datetime.now()
    print("begin time: ", begin_time)
    device_df_list = split_df(df, n_part)

    p = mp.Pool(n_part)
    for i in range(n_part):
        df_part = device_df_list[i].copy()
        print("number {} part is processing".format(i))
        try:
            p.apply_async(create_daily, args=(df_part, ), callback=log_result)
        except Exception as e:
            print("something went wrong")
            print(e)

    p.close()
    p.join()
    end_time = datetime.now()
    print("end time: ", end_time)
    print("time consumed: ", end_time - begin_time)
    return result_list

"""
uuid generation 
"""
from faker import Faker 
f1 = Faker()
seed = 99
f1.seed(seed)
my_uuid = f1.uuid4()
seed += 1

import uuid 
cid = uuid.uuid4().hex