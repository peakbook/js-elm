Basic ELM for javascript
========================

## Requirements
- [Numeric Javascript](http://www.numericjs.com/)

## Example
Classification
```js
var d = 2;        // the dimension of input data
var nclass = 2;   // the number of classes 
var L = 10;       // the number of hidden nodes
var alpha = 1e-8; // L2 regularization coefficient
var X = [[0,0],[0,1],[1,0],[1,1]];
var T = [[0],[1],[1],[0]];
var elm = new ELM(d, L, nclass, alpha, MappingFuncs.Sigmoid, ELMType.Classification);

elm.train(X, T);
Y = elm.predict(X);
```

This `Basic ELM` provides the following Mapping Functions.
- `MappingFuncs.Sigmoid`
- `MappingFuncs.HyperbolicTangent`
- `MappingFuncs.Gaussian`
- `MappingFuncs.Multiquadric`
- `MappingFuncs.HardLimit`
- `MappingFuncs.Cosine`

You can also choose regression or classification.
- `ELMType.Regression` for regression
- `ELMType.Classification` for Classification

