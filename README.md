# bayer-filter-rgb-reconstruction
Rekonstrukcija RGB signala iz Bayerova filtra u digitalnim kamerama.

Preduvjeti

* Instaliran Python 3.x (preporuka: 3.10+)

* Instaliran Git (za kloniranje repozitorija)
  


**Preuzimanje projekta (kloniranje repozitorija)**

U terminalu se repozitorij preuzima naredbom:
```
git clone bayer-filter-rgb-reconstruction

cd bayer-filter-rgb-reconstruction
```
**Instalacija ovisnosti** 

Svi potrebni Python paketi mogu se instalirati globalno pomoću `pip` alata:
```
pip install pillow opencv-python numpy scipy scikit-image sewar matplotlib tqdm torch
```


**Pokretanje aplikacije** 

GUI aplikacija se pokreće iz glavnog direktorija repozitorija sljedećom naredbom: 

```python -m gui.gui```
