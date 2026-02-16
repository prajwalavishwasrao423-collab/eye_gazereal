import requests

ESP_IP = "192.168.1.100"  # change if needed

def send_to_esp32(action):
    url = f"http://{ESP_IP}/control?action={action}"
    print("📡 Sending to ESP:", url)
    requests.get(url, timeout=1)




