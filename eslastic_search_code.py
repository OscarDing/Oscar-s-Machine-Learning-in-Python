def get_es_total_tbox(deviceno='868120223922664'):
    # 取总数
    es = Elasticsearch(hosts="192.168.1.174")
    body_ = {
        "query": {
            "bool": {
                "filter": [{
                    "bool": {
                        "must": [
                            {
                                "match_phrase": {
                                    "DEVICENO": {
                                        "query": deviceno
                                    },

                                }
                            },
                            {
                                "match_phrase": {
                                    "ACCSTATE": {
                                        "query": 10
                                    },

                                }
                            },

                        ]
                    }
                },

                ]
            }
        },
        "sort": [
            {"CREATEDATE": "desc"}
        ],
    }
    b = es.search(index="sl_gpsdata_index", body=body_,request_timeout=300)['hits']["total"]
    return b


def get_es_data_tbox(page, deviceno, date_gte, date_lte):
    """
    对于一辆车, 取出es里的相关数据
    :param page:
    :param deviceno:
    :return:
    """
    es = Elasticsearch(hosts="192.168.1.174")
    # 改造 需要搜寻时间和车牌号
    # ACCSTATE: 车辆状态 KM: 公里  SOC: 电量
    body_ = {
        "query": {
            "bool": {
                "filter": [
                    {
                    "bool": {
                        "must": [
                            {
                                "match_phrase": {
                                    "DEVICENO": {
                                        "query": deviceno
                                    }
                                }
                            },
                            {
                                "match_phrase": {
                                    "ACCSTATE": {
                                        "query": 10
                                    },

                                }
                            },
                        ]
                    }
                    },
                    {
                        "range": {
                            "CREATEDATE": {
                                "gte": date_gte,
                                "lte": date_lte
                            }
                        }
                    }
                ]
            }
        },
        "size": 5,
        "from": page,
        "sort": [
            {"CREATEDATE": "desc"}
        ],
        "_source": ["ACCSTATE", "CREATEDATE", "KM", "SOC",  # "RECIVETIME",  "SPEED",
                    "LOCATION"]
    }

    b = es.search(index="sl_gpsdata_index", body=body_, request_timeout=300)
    return b["hits"]['hits']



def es_to_df(es_list):
    """
    把从es取来的list转化成df
    :param es_list:
    :return: df
    """
    df = pd.DataFrame()
    for one in es_list:
        loc = one['_source']['LOCATION']
        loc = [round(float(x), 5) for x in loc.split(",")]
        km = one['_source']['KM']
        soc = one['_source']['SOC']
        createdate = one['_source']['CREATEDATE']
        createdate = datetransform(createdate)
        df_row_dict = {"location": loc, "km": km, "soc": soc, "createdate": createdate}

        df = df.append(df_row_dict, ignore_index=True)

    return df


