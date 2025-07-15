def get_current_light_state(time_in_s):
    # 可设置断点调试逻辑
    if time_in_s % 10 < 5:
        return [{"color": "green"}]
    else:
        return [{"color": "red"}]
