from selenium import webdriver
import chromedriver_autoinstaller
import getpass
import os
import time
import requests
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0]
print(chrome_ver)
driver_path = f'./{chrome_ver}/chromedriver'
if os.path.exists(driver_path):
    print(f"chromedriver is installed: {driver_path}")
else:
    print(f"Install the chromedriver (version: {chrome_ver})")
    chromedriver_autoinstaller.install(True)

search = input("Enter the search keyword: ")
password = getpass.getpass("Enter your password: ")

driver = webdriver.Chrome() # driver_path
driver.get(f"https://www.pinterest.co.kr/search/pins/?q={search}")
driver.implicitly_wait(3) 
# login_button = driver.find_element_by_xpath('//*[@id="__PWS_ROOT__"]/div/div[1]/div/div[1]/div/div[3]/div[1]/button/div')
login_button = driver.find_element(By.XPATH, '//*[@id="__PWS_ROOT__"]/div/div[1]/div/div[1]/div/div[3]/div[1]/button/div')
login_button.click()

# input_ID_element = driver.find_element_by_xpath('//*[@id="email"]')
input_ID_element = driver.find_element(By.XPATH, '//*[@id="email"]')
input_ID_element.send_keys()  # Enter your ID

# input_PS_element = driver.find_element_by_xpath('//*[@id="password"]')
input_PS_element = driver.find_element(By.XPATH, '//*[@id="password"]')
input_PS_element.send_keys() # Enter your PWD

# login_button = driver.find_element_by_class_name('SignupButton')
login_button = driver.find_element(By.CLASS_NAME, 'SignupButton')
login_button.click()
time.sleep(5)
last_height = driver.execute_script("return document.body.scrollHeight")
image_count = 1 

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    new_height = driver.execute_script("return document.body.scrollHeight")

    if new_height == last_height:
        break

    last_height = new_height
    # image_elements = driver.find_elements_by_css_selector('img')
    image_elements = driver.find_elements(By.CSS_SELECTOR, 'img')

    os.makedirs(search, exist_ok=True)

    for i, image_element in enumerate(image_elements):
        image_url = image_element.get_attribute('src')

        if image_url and image_url.startswith('https://'):

            response = requests.get(image_url, stream=True)

            file_name = f"{search}/image_{image_count}.jpg"

            with open(file_name, 'wb') as file:
                file.write(response.content)

            print(f"Image {image_count} saved successfully.")

            with open(f"{search}/links.txt", "a") as file:
                file.write(f"Image {image_count}: {image_url}\n")

            image_count += 1

driver.quit()
