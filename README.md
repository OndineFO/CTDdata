# CTDdata
Hetta forrit kann nýtast at tekna CTD dátur við

Við hesum forriti fært tú yvirlit yvir CTD data frá mátitúrum.

Forritið spyr um eitt túrnummar (TTTT) og fer síðani í viðkomandi mappu eftir data og teknar profilar fyri nakrar fyriætlaðar eginleikar.

Fyri at skilja forritið, má ein skilja dátubygnaðin (mynd niðanfyri).
Dátusettini, sum vit hava áhuga í, liggja í havTTTT mappuni, tvær fílur fyri svørja støð (máting) báðar nevndar við støðnummari td 15850001 og 15850001Ox. Hesum sleppa vit lætt at, við at hyggja ígjøgnum kunningarfíluna hydrTTTT


Við túrnummarinum frá brúkaranum klára vit at koma inn í rætta cruTTTT mappu. Har liggur ein fíla hydrTTTT við støðnummari, degi og tíð, og longd og breidd.

Í CTD er ein mappa fyri hvønn túr cruTTTT.
Í hvørjari cruTTT mappu eru kunningarfílur og ein dátumappa havTTTT.
Í havTTTT er data fyri hvørja støð (máting) nevnd við støðnummarinum. 


Mappubygnaður: 
                                        |--- hydr1585.dat
                           |--cru1585-- |--- stodfil1585.csv    |- 15850001.raw, 15850002.raw ...
                           |            |--- hav1585 ---------- |- 
            |--DATA---CTD--|                                    |- 15850001Ox.raw, 15850002Ox.raw ...
            |              |            |--- hydr1461.dat
    WORK--- |              |--cru1461-- |--- stodfil1461.csv    
            |                           |--- hav1461 ---------- |- Same as above
            |--PROG---TurYvirlit.py                             
            
            
            

Eftir sum forritið leypur ímillum mappur er umráðandi at mappubygnaðurin er sum á myndini omanfyri.
Forritið skal liggja í eini mappu (td PROG), ið er granni hjá DATA mappuni. Í DATA er mappan CTD og í henni allar cruTTTT mappurnar. Hvør cruTTTT mappa skal millum annað innihalda mappuna havTTTT og fílurnar hydrTTTT.dat og stodfilTTTT.csv
