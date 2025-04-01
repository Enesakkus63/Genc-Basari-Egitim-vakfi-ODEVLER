from requests import get
from pprint import pprint
endpoint= 'https://newsapi.org/v2/everything?domains=wsj.com&apiKey=d1a3b359dc344a5dafa3eec0233dadfd'
response = get(url=endpoint)
data= response.json()
articles = data['articles']
def read_articles():
    for article in articles:
        print(f"Başlık: {article['title']}")
        print(f"Açıklama: {article['description']}")
        print(f"Yayınlanma Tarihi: {article['publishedAt']}")
        print("-" * 50)
def create_article(new_article):
    articles.append(new_article)
    print("Yeni haber eklendi!")
def update_article(index, updated_data):

    if 0 <= index < len(articles):
        articles[index].update(updated_data)
        print(f"Haber {index} güncellendi!")
    else:
        print("Geçersiz haber index'i!")
def delete_article(index):

    if 0 <= index < len(articles):
        articles.pop(index)
        print(f"Haber {index} silindi!")
    else:
        print("Geçersiz haber index'i!")


new_article = {
    'source': {'id': 'example', 'name': 'Example Source'},
    'author': 'Jane Doe',
    'title': 'Yeni Başlık',
    'description': 'Yeni açıklama',
    'url': 'https://example.com',
    'publishedAt': '2025-02-28T10:00:00Z',
    'content': 'İçerik burada yer alır'
}
create_article(new_article)
read_articles()
