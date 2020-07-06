from selenium import webdriver
from time import sleep


class InstaBot:
    def __init__(self, username, pw):
        self.msgnum = 19
        self.driver = webdriver.Chrome()
        self.username = username
        self.driver.get("https://instagram.com")
        sleep(2)
        self.driver.find_element_by_xpath("//input[@name=\"username\"]")\
            .send_keys(username)
        self.driver.find_element_by_xpath("//input[@name=\"password\"]")\
            .send_keys(pw)
        self.driver.find_element_by_xpath('//button[@type="submit"]')\
            .click()
        sleep(4)
        self.driver.find_element_by_xpath("//button[contains(text(), 'Not Now')]")\
            .click()
        sleep(2)

    def get_unfollowers(self):
        self.driver.find_element_by_xpath("//a[contains(@href,'/{}')]".format(self.username))\
            .click()
        sleep(2)
        self.driver.find_element_by_xpath("//a[contains(@href,'/following')]")\
            .click()
        following = self._get_names()
        self.driver.find_element_by_xpath("//a[contains(@href,'/followers')]")\
            .click()
        followers = self._get_names()
        not_following_back = [user for user in following if user not in followers]
        print("LIST OF NON FOLLOWERS: " + str(not_following_back))

    def _get_names(self):
        sleep(2)
        #sugs = self.driver.find_element_by_xpath('//h4[contains(text(), Suggestions)]')
        #self.driver.execute_script('arguments[0].scrollIntoView()', sugs)
        sleep(2)
        scroll_box = self.driver.find_element_by_xpath("/html/body/div[4]/div/div[2]/ul")
        last_ht, ht = 0, 1
        # while last_ht != ht:
        #     last_ht = ht
        #     sleep(1)
        #     ht = self.driver.execute_script("""
        #         arguments[0].scrollTo(0, arguments[0].scrollHeight); 
        #         return arguments[0].scrollHeight;
        #         """, scroll_box)
        fBody = self.driver.find_element_by_xpath("//div[@class='isgrP']")
        #scroll = 0
        for i in range(1,6):
            self.driver.execute_script('arguments[0].scrollTop = arguments[0].scrollTop + arguments[0].offsetHeight;', fBody)
            sleep(0.1)
        links = scroll_box.find_elements_by_tag_name('a')
        names = [name.text for name in links if name.text != '']
        # close button
        self.driver.find_element_by_xpath("/html/body/div[4]/div/div[1]/div/div[2]/button")\
            .click()
        #print(names)
        return names
    def dmFromHome(self, name):
        self.driver.find_element_by_xpath('//*[@id="react-root"]/section/nav/div[2]/div/div/div[3]/div/div[2]/a')\
            .click()
        sleep(2)
        self.driver.find_element_by_xpath("//a[@class='-qQT3 rOtsg']")\
            .click()
        #text = self.driver.find_element_by_xpath('//*[@id="react-root"]/section/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div/div/div[2]/div[2]/div/div/div/div/div/div/div/div/span').text
    def getText(self, num):
        input = ""
        try:
            input = self.driver.find_element_by_xpath('//*[@id="react-root"]/section/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div/div/div[{}]/div[2]/div/div/div/div/div/div/div/div/span'.format(str(num))).text
        except:
            print("no message")
            input = ""
        return input
    def sendMessage(self, msg):
        self.driver.find_element_by_xpath('//*[@id="react-root"]/section/div/div[2]/div/div/div[2]/div[2]/div/div[2]/div/div/div[2]/textarea').send_keys(msg)
        sleep(1)
        self.driver.find_element_by_xpath('//*[@id="react-root"]/section/div/div[2]/div/div/div[2]/div[2]/div/div[2]/div/div/div[3]/button').click()
        self.msgnum += 2
        

user = ""
pw = ""
name = ""
my_bot = InstaBot(user, pw)
sleep(2)
my_bot.dmFromHome(name)
inp = my_bot.getText(my_bot.msgnum)
#print(my_bot.getText(19))




#//*[@id="react-root"]/section/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div/div/div[2]/div[2]/div/div/div/div/div/div/div/div/span
#//*[@id="react-root"]/section/div/div[2]/div/div/div[2]/div[2]/div/div[1]/div/div/div[19]/div[2]/div/div/div/div/div/div/div/div/span

