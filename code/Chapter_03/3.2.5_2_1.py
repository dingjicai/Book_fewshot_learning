#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import json

API_KEY = "???"
SECRET_KEY = "????"

def main():
    url = "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/txt2img?access_token=" + get_access_token()

    payload = json.dumps({
        "text": "千山鸟飞绝,万径人踪灭,孤舟蓑笠翁,独钓寒江雪",
        "resolution": "1024*1024",
        "style": "写实风格"
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

if __name__ == '__main__':
    main()