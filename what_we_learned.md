
### durante la cross validation di monk3
- il momentum velocizza di molto la convergenza nelle fasi iniziali, e in generale nei plateau
- l'approccio stocastico (online/minibatch) ha un effetto regolarizzante.
- tychonov, con online e spiccoli mb, ha un effetto maggiore, perciò può portare all'underfitting. Bisogna diminuire lambda se si diminuisce mb
 anche per valori di lambda che invece sono adatti in batch
- il momentum può smorzare l'effetto regolarizzante (e underfittante) del approccio stocastico
- la regolarizzazione può aiutare la convergenza (?)

### grid search tanh
- per la tanh 1 di eta è troppo alto, anche se il linear decay lo decima
- il momentum rende instabile
- la lambda rende la curva iniziale più smooth
- con l'aumento della complessità della rete, aumenta pure l'importanza di lambda e alpha