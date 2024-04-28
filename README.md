# iremSezerNotebook

akıl yürütme sırasında elde edilen dış bilgiyi kullanmanın mümkün olduğu düşünce zinciri (CoT)

## Python
sil!
![not1](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/83e9c55d-b637-47f0-a695-4458db7f9b7c)

# List methods & functions Sil

planets.append('Pluto')

planets.pop()

Searching lists
planets.index('Earth')

sorted(planets) #sıralar

# List comprehensions

loud_short_planets = [planet.upper() + '!' for planet in planets if len(planet) < 6]
loud_short_planets

# Dictionaries Sil

numbers = {'one':1, 'two':2, 'three':3}
örnek makine öğrenmesi kodu
# Underfitting and Overfitting
Modelimizi eğitim verilerinden elde edilen örüntülere göre oluşturuyoruz. Bu işlem sonucunda iki şeyden biri olabilir; modelimiz aşırı öğrenebilir veya eksik öğrenebilir. Bu durumda modelimiz yeterli öngörüde bulunamayacak ve tahminlerimizde hata oranı yüksek olacaktır.

Modelimiz, eğitim için kullandığımız veri setimiz üzerinde gereğinden fazla çalışıp ezber yapmaya başlamışsa ya da eğitim setimiz tek düze ise overfitting olma riski büyük demektir.

bir model yetersiz öğrenmeye sahipse, modelin eğitim verilerine uymadığı ve bu nedenle verilerdeki trendleri kaçırdığı anlamına gelir. Ayrıca modelin yeni veriler için genelleştirilemediği anlamına da gelir.
## Pandas

# iloc & loc

loc komutu ile etiket kullananarak verimize ulaşırken, iloc komutunda satır ve sütün index numarası ile verilerimize ulaşmaktayız, Yani loc komutunu kullanırken satır yada kolon ismi belirtirken, iloc komutunda satır yada sütünün index numarasını belirtiyoruz.

## Feature engineering 

Feature engineering (Özellik Mühendisliği), ham verileri seçme, değiştirme ve denetimli öğrenmede kullanılabilecek özelliklere dönüştürme ve veriyi makine öğrenmesi modellerine hazırlama sürecidir. 

Makine öğrenimi algoritmalarının daha iyi performans göstermesine yardımcı olan özellikler veya girdi değişkenleri oluşturmak için alan bilgisini kullanma sürecidir.

modelin tahmin gücünü arttırmaya yardımcı olur.

# Mutual Information

from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores

# Outlier Detection - aykırı gözlem


!(https://www.veribilimiokulu.com/wp-content/uploads/2019/12/image-1.png)





