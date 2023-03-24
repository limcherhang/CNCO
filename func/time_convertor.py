def convert_sec(times):
    if times < 60:
        result = f"{round(times)} sec"
    elif times < 3600:
        m = times // 60
        s = round(times - m * 60)
        result = f"{m} minutes {s} sec"
    elif times < 86400:
        h = times // 3600
        s = times - h * 3600
        m = s // 60
        s = round(s - m * 60)
        result = f"{h} hour {m} minutes {s} sec"
    else:
        d = times // 86400
        s = times - d * 86400
        h = s // 3600
        s = s - h * 3600
        m = s // 60
        s = round(s - m * 60)
        result = f"{d} day {h} hour {m} minutes {s} sec"
    return result
