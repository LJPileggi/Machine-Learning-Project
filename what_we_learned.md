
### durante la cross validation di monk3
- il momentum velocizza di molto la convergenza nelle fasi iniziali, e in generale nei plateau
- l'approccio stocastico (online/minibatch) ha un effetto regolarizzante.
- tychonov, con online e spiccoli mb, ha un effetto maggiore, perciò può portare all'underfitting. Bisogna diminuire lambda se si diminuisce mb
 anche per valori di lambda che invece sono adatti in batch
- il momentum può smorzare l'effetto regolarizzante (e underfittante) del approccio stocastico
- la regolarizzazione può aiutare la convergenza (?)