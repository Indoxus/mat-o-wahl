import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep

totalthesis = 38

def fill_out(questions,multiplier):
    driver = webdriver.Chrome()
    driver.get("https://www.wahl-o-mat.de/bundestagswahl2025/app/main_app.html")
    sleep(2)
    driver.find_element(By.XPATH,"/html/body/div[1]/div/main/section/div/div[1]/button").click()
    sleep(1)
    for i in range(totalthesis):
        if questions[i] != 255:
            driver.find_element(By.XPATH,"/html/body/div[1]/div/main/section/div/div/div[1]/ol/li["+str(i+1)+"]/div[2]/div/div/div[2]/ul/li["+str(3-questions[i])+"]/button").click() #frage 1, v = 2
        else:
            driver.find_element(By.XPATH,"/html/body/div[1]/div/main/section/div/div/div[1]/ol/li["+str(i+1)+"]/div[2]/div/div/div[2]/div/button").click()
        sleep(0.4)
    for i in range(totalthesis):
        if multiplier[i]:
            target_element = driver.find_element(By.XPATH,"/html/body/div[1]/div/main/section/form/div/ol/li["+str(i+1)+"]/div[1]/div/label")
            driver.execute_script("arguments[0].scrollIntoView();", target_element)
            target_element.click()
    parteienbutton = driver.find_element(By.XPATH,"/html/body/div[1]/div/main/div[2]/button[2]")
    driver.execute_script("arguments[0].scrollIntoView();", parteienbutton)
    parteienbutton.click()
    alleparteien = driver.find_element(By.XPATH,"/html/body/div[1]/div/main/form/section/div[3]/div[1]/label")
    driver.execute_script("arguments[0].scrollIntoView();", alleparteien)
    alleparteien.click()
    ergebnis = driver.find_element(By.XPATH,"/html/body/div[1]/div/main/form/section/div[3]/div[3]/button[2]")
    driver.execute_script("arguments[0].scrollIntoView();", ergebnis)
    ergebnis.click()
    
    sleep(30)
