# iremSezerNotebook

akıl yürütme sırasında elde edilen dış bilgiyi kullanmanın mümkün olduğu düşünce zinciri (CoT)

## Python

# List comprehensions

loud_short_planets = [planet.upper() + '!' for planet in planets if len(planet) < 6]
loud_short_planets

# Underfitting & Overfitting

![uo](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/29de5db8-3d00-4b2a-b998-5a7a7fe43a12)


Modelimizi eğitim verilerinden elde edilen örüntülere göre oluşturuyoruz. Bu işlem sonucunda iki şeyden biri olabilir; modelimiz aşırı öğrenebilir veya eksik öğrenebilir. Bu durumda modelimiz yeterli öngörüde bulunamayacak ve tahminlerimizde hata oranı yüksek olacaktır.

Modelimiz, eğitim için kullandığımız veri setimiz üzerinde gereğinden fazla çalışıp ezber yapmaya başlamışsa ya da eğitim setimiz tek düze ise overfitting olma riski büyük demektir.

bir model yetersiz öğrenmeye sahipse, modelin eğitim verilerine uymadığı ve bu nedenle verilerdeki trendleri kaçırdığı anlamına gelir. Ayrıca modelin yeni veriler için genelleştirilemediği anlamına da gelir.

## Pandas

# iloc & loc

loc komutu ile etiket kullananarak verimize ulaşırken, iloc komutunda satır ve sütün index numarası ile verilerimize ulaşmaktayız, Yani loc komutunu kullanırken satır yada kolon ismi belirtirken, iloc komutunda satır yada sütünün index numarasını belirtiyoruz.

## Feature Engineering 

![1666105956546](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/b819196a-316e-4802-a2ac-b0b190d33901)

Feature engineering (Özellik Mühendisliği), ham verileri seçme, değiştirme ve denetimli öğrenmede kullanılabilecek özelliklere dönüştürme ve veriyi makine öğrenmesi modellerine hazırlama sürecidir. 

Makine öğrenimi algoritmalarının daha iyi performans göstermesine yardımcı olan özellikler veya girdi değişkenleri oluşturmak için alan bilgisini kullanma sürecidir.

modelin tahmin gücünü arttırmaya yardımcı olur.

# Mutual Information
Mutual Information (Ortaklı Bilgi) Yöntemi

Bilgi teorisi alanından gelen ortaklı bilgi (mutual information), bilgi kazanımının (tipik olarak karar ağaçlarının yapımında kullanılır) özellik seçimine uygulanmasıdır.

Ortaklı bilgi iki değişken arasında hesaplanır ve diğer değişkenin bilinen bir değeri verildiğinde bir değişken için belirsizlikteki azalmayı ölçer.

kod
# import the required functions and object.
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

# Kalmasını istediğiniz değişken sayısı
select_k = 10

# Değişkenlerin seçim stratejisi belirlenir
# mutual_info_classif = ortalı bilgi yönteminin kullanılmasıdır
selection = SelectKBest(mutual_info_classif, k=select_k).fit(x_train, y_train)

# İlişkili olan değişkenler gösterilir.
features = x_train.columns[selection.get_support()]
print(features)

---------------
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores

# Outlier Detection
Aykırı Gözlem


![not22](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/6cb3dd3c-0802-4e52-9536-395c81fdf27c)

# K-Means Algoritması

![k1](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/dbd81a8d-94f8-4d61-b5c4-e1b370f4b821)

benzer nesneleri otomatik olarak aynı gruplara gruplandırır.

Farklı kümelerdeki veri noktaları çok farklıyken, aynı alt gruptaki (küme) veri noktalarının çok benzer olması nedeniyle verilerdeki alt grupların belirlenmesi görevi olarak tanımlanabilir. 

unsupervised learning(gözetimsiz öğrenme) ve kümeleme algoritmasıdır.

--kod
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

X.head()

sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
);
![k2](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/aa4c4326-48b9-4a25-a42f-26e0e16d4b11)

## Deep Leaarning

Derin Öğrenme bir makine öğrenme yöntemidir. 
Verilen bir veri kümesi ile çıktıları tahmin edecek yapay zekayı eğitmemize olanak sağlar. 
Yapay zekayı eğitmek için hem denetimli hem de denetimsiz öğrenme kullanılabilir.

## Data Cleaning

scaling (ölçeklendirme) verilerinizin aralığını değiştirirken,
normalization (normalleştirme) verilerinizin dağıtım şeklini değiştiriyorsunuz.

--kod
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size=1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

# plot both together to compare
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
ax[1].set_title("Scaled data")
plt.show()
---

![s1](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/4439facc-867d-48f1-9701-ade0d09c1956)

Normalization (normalleştirme) verilerinizin dağıtım şeklini değiştiriyorsunuz.
kod

---
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(normalized_data[0], ax=ax[1], kde=True, legend=False)
ax[1].set_title("Normalized data")
plt.show()
---

![norm](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/825475f0-52d2-4051-b527-58f4eb1160be)



