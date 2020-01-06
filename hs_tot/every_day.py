# 获取两个日期间的所有日期
import datetime

def getEveryDay(begin_date, end_date):
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    # while begin_date <= end_date:
    #     date_str = begin_date.strftime("%Y-%m-%d")
    #     date_list.append(date_str)
    #     begin_date += datetime.timedelta(days=1)
    # 返回两个变量相差的值，就是相差天数
    total_days = (end_date - begin_date).days  # 将天数转成int型

    # date_temp = datetime.datetime.strptime("2019-10-10", '%Y-%m-%d').date()
    # date_temp1 = datetime.datetime.strptime("2019-10-12", '%Y-%m-%d').date()
    # print(date_temp>date_temp1 )
    # print(type(date_temp))
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=total_days / 14)
    return date_list


# print(list(set(getEveryDay('2016-01-01', '2016-01-08'))))
#去重
# every_day = getEveryDay('2016-01-01', '2016-01-09')
# set_day_list = []
# for day in every_day:
#     if day not in set_day_list:
#         set_day_list.append(day)
# print(set_day_list)
# print(getEveryDay('2016-01-01', '2016-02-01'))
