import urllib
import json
from bs4 import BeautifulSoup
def load_url(url):
    print("Loading url = ",url)
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page,"html.parser")
    script = soup.find_all("script",attrs={"type": "application/ld+json"})[0]
    script = str(script)
    script = script.replace("</script>","").replace("<script type=\"application/ld+json\">","")
    csvdata = []
    for element in json.loads(script)["review"]:
        if "reviewBody" in element:
            csvdata.append([element["reviewBody"]])
    return csvdata
if __name__ =="__main__":
    a= load_url("https://www.lazada.vn/products/kinh-cuong-luc-10d-cho-iphone-6plus6splus66s7-8-7plus-8plus-x-xs-xsmax-1111-pro-11-pro-max-sieu-ben-phu-kien-dien-thoai-i276346806-s425928695.html")
    print(a)