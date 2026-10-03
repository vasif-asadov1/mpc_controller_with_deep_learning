# Thesis Workflow


Thesis ismi "Deep Learning based Model Predictive Control of nonlinear systems" - dir. Tezimizin maksadı Deep Learning ve MPC algoritmaları kullanarak nonlinear sistemleri real time zamanlı kontroletmektir. Nonlinear sistem olarak inverted cart pendulum sistemini ele aldım. Bu sistemin, yani ters sarkacın differential equationlar ile verilmiş hareket fiziğini kullanarak gerçekci bir veri seti üretmemiz lazım. Bu veri setini üretirken aşağıdaki bilgileri göz önünde bulundurmak zorundayız: 


# Part 1. Dataset 

1. Ters sarkaç modelinde sarkaç cart - yani arabanın yan tarafına doğru monte edildiği için, 360 derece boyunca serbest dönebilir. Yani, sarkaç yerle temas etmiyor. Sarkacı yere kiyasla 90 derecelik açıda dik  şekliyle bırakırsak, ağırlık kuvvetinin altında hareket ederek en son tam aşağıya, 180 derecelık açıya ulaşır. Yani yere perpendicular şekilde kalır. Yani sarkacın hareketi 360 derecelik bir açıdadır ve yere temas etmez.
2. İnput değerler, x, x_dot, theta, theta_dot ve input force-dur. Biz veri seti uretirken bir sonraki state degerlerini bulacağız. [-1, 1] bandında Force uygulayacağız. Bu Force negative olursa sola doğru, positive olursa sağaa doğrudur. Her bir adımda force kullanmak gerekmiyor. Çünkü bizim veri setimiz uzun süre force kullanılmadığında sistem nasıl tepki verecektir, onu da bilmek zorunda. Yani sadece ağırlık kuvvetinin altında sistemin nasıl hareket ettiğini gözlemleyebilmeliyiz. Bunun için de, her saniye force uygulamak yerine belli aralıklarda uygulayıp, bazen uygulamamak gibi senarilere ihtiyacımız var ki, sistemin tepkiselliğini ve fizik kurallarını deep learning ile gerçeğe en yakın şekilde öğrenebilelim.
3. Output değerler, next state-ler, yani, `[x_n, x_dot_n, theta_n, theta_dot_n]` - dir. Biz daha accurate olsun diye Runge Kutta 4 metodunu kullanabiliriz. 10,000 tane farklı başlangıç noktalarından başlayarak, 1000 adım ilerleyen bir veri seti kurup, sonda 10,000,000-luk bir veri seti elde etmemiz hedefleniyor. 
4. 10 milyonluk veri setini parquet dosyasi olarak save etmek lazim. Bu dosyayi save ettikten sonra Deep Learning aşamasına geçebiliriz. 


# Part 2. Deep Learning

Veri setimizi parquet dosyasi olarak save ettikten sonraki adım Deep Learning ile modellemektir. Veri setimiz gerçek bir ters sarkac modelinin gercek hayattaki fiziksel hareketlerini ve dinamigini temsil ettiği için, train edeceğimiz model de sonda ters sarkaç modelimizin deep learning ile train edilmiş hali olacaktır. Trainingin ne kadar başarılı olduğunu test etmemiz şarttır. Bu test için ise multi-step rollout simulasyonu yapmamiz gerekiyor. Multi-step rollout simulasyonunda iki tane şeyi kıyaslayacağız. 

1. Gerçek sarkaç pendulumun değerlerini kullanarak Verilmiş herhangi başlangıç force altındaki hareketlerini izleyeceğiz. Bu bize true value verecektir. 
2. Eğitiilmiş modelimizin hemen force değeri altındaki hareketlerini izleyeceğiz. Her adımda model kendi sonucunu, bir sonraki step-in başlangıcı olarak kabuledecektir. Buyüzden eğer herhangi bir yerde en ufak bir sapma olursa, her bir sonraki stepde bu sapma daha fazla büyüycektir. 10, 20, 30, 40 saniyelik süreler boyunca bu sapmaların nekadar true valuedan uzaklaştığını ölçeceğiz. hem True hem de Predicted valueları aynı grafiklerde vererek bunu simule edeceğiz. 
3. Sadece ağırlıl kuvetinin altında sistemin nasıl tepki verdiğini gözlemleyeceğiz. İnverted pendulum tam dikey açıda bırakılacaktır ve ağırlık kuvetinin altında bırakılacaktır. Sallana-sallana en son pozisyona kadar gelip duracaktır. Bu simülasyonda state değerlerimizin nasıl değiştiğini gözlemleyeceğiz.
4. Diğer simülasyonlarda ise farklı önemli caseleri simüle edeceğiz. Farklı başlangıç noktalarından bırakılma,  farklı aralıklarla force uygulanma gibi birkaç önemli anları simule edip, pendulumun hareketlerini modelimizin hareketleri ile kıyaslayacağız. 

Modelimizin nekadar iyi olduğunu ölçmek için gereken bazı başka simulasyonlar varsa eğer, bunları kendin de bana tavsiye edebilirsin. Her bir simulasyon zamanı nelerin yanlış gittiğini, sapmaların neden büyüdüğünü tespit edip, veri seti üretme aşamasına geri dönebiliriz. Ta ki, sapmaların en az olduğu simulasyonu elde edene kadar. 

Sapmaların en az olması için gereken trained modeli elde etmemiz için neural ağlardaki nöron sayılarını yüksek tutmamız gerekebilir. Ben 2 veya en fazla 3 hidden katman düşünüyorum. Ama ideal halini sen önerirsin. Her bir katmandaki nöron sayısını, çok fazla tutmamak lazım ki, computation fazla vaktimizi almasın. Ayrıca laptop hardware kapasitem aşılmasın. Buyüzden de sweet spot u belirlememiz çok önemlidir. Ayrıca en önemli kural, aktivasyon fonksiyonu olarak hidden layerlerde ReLu aktivasyon fonksiyonunu kullanmaktır. Çünkü biz sonradan bu ReLu fonksiyonlarını kullanarak hidden layer çıktılarını kontroletmeye çalışacağız.

Sapmaların en az olduğu modelin trainingini bulduktan sonra hafif bir hyperparemeter tuning de yapmamiz gerekiyor. Hyperparameter tuning bize kullandigmiz noron sayisinin idealitesini verify edecektir ki, bu da tez için olmazsa olmazdır. 

Neural networks katmanlarımızda ReLu aktivasyon fonksiyonu kullanmamız şart. Çünkü daha sonra biz bunu MİLP (mixed integer linear programming) kullanarak COST Function minimize etme çabalarına gireceğiz. 

Buraya kadarki süreci az-çok biliyorum. Ama burdan sonraki süreçle alakalı çok fikrim yok. Bundan sonraki süreci senin rehberligine bırakmam gerekiyor. Ben sadece bilgileri veriyorum.


# Part 3. MILP & MPC

Bu aşamaya gelene kadar elimizde şunlar olacaktır:

1. 10 milyonluk devasa ve detaylı bir veri seti. 
2. İnverted Cart Pendulum sisteminin dinamiklerinin üzerinde train edilmiş, gerçek fizik kurallarını en az error ile yansıtan ReLu aktivasyon fonksiyonlu 2 (en fazla 3) hidden layerli neural networks modeli.
3. Deep Learning modelimiz ile gercek inverted pendulum sisteminin (predicted values vs actual values) simulasyon kıyasları. Multi-step rollout simulasyonları ile yanaşı diğer gereken simulasyonlar (bunları sen belirleyeceksin).

Dördüncü aşama ise Mixed İnteger Linear Programming kullanarak MPC control uygulamak. Burada gurobipy kullanacağız. Academic License-a sahibim. Bizim bu aşamada maksadımız Cost Function veya Optimization Function oluşturmaktır. Bu fonksiyonu MİLP kuralları ile çözmek ve MPC uygulamaktır. En ama en önemli aşama constraintleri, Optimization Fonksiyonunu, MPC sorununu net ve sorunsuz şekilde oluşturmaktır. Bu aşamada herşeyin pürüzsüz olduğunu, doğru olduğunu defalarca kontroletmemiz gerekiyor. Control aşamasına geçmeden önce formüllerimizin dosdoğru olduğundan, hangi değişkenin neyi gösterdiğinden emin olmak zorundayız.

Elimde inverted pendulum sistemiyle alakalı yazılmış bir paper var. Gerektiğinde bu paperde yazılanları incelememiz gerekecektir. Buradaki formüllerin neleri gösterdigini bilmemiz gerkiyor. Ve kendi paperimiz ile bu paper arasındaki farkları net şekilde belirlememiz gerekiyor. Bu belirlemeler olmadan ilerlemek büyük bir risk taşır. 

Kısacası, Part 3, Fonksiyonlar, Denklemler oluşturacağımız, doğruluklarını kontroledeceğimiz, Deep Learning modelimizi Fonksiyonun içine gömmeye çalışacağımız bir aşama olacaktır. Bu aşamada büyük titizlik yapmamız, herşeyin doğru olduğunu check etmemiz gerekiyor. Bu check & verification olmazsa ilerlemek makul değildir.


# Part 4. Simulations

Part 4 ise noktayı koyduğumuz stepdir. Oluşturduğumuz formüllerimizi, fonksiyonlarımızı ele alıp, MPC controllere sokacağız. Controlun nasıl ilerledğini adım-adım takip etmemiz gerekecektir. Simulasyonlar, verificationlar, zaman takipi, real-time application için uygunluğu bu bölümde simule edilecek ve tartışılacaktır. Elimizde bir çok case için ve farklı horizonlar için MPC control simulasyonları toplamamız gerekiyor. 

# İmprovements

Simulasyon sonuçlarına göre gereken improvementler belirlenecektir. Bu improvementler bizi hem yazilan paperden farklandiracak, hem kendi imzamizi yazdirmamiza vesile olacaktir. Real time kontrolü hızlandırmak için birtakım fikirler sunacağım sana. Sen de onları değerlendireceksin. Kendin daha iyi yöntem ve metotlar belirleyeceksin. Son olarak, belirlediğimiz metotları uygulayacağız. 
