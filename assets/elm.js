// numeric.js(http://www.numericjs.com/) is required 

numeric.zeros = function(sz)
{
    var x = numeric.random(sz);
    for(var j=0;j<sz[0];j++)
    {
        for(var i=0;i<sz[1];i++)
        {
            x[j][i] = 0.0;
        }
    }
    return x;
}

var ELMType = {
    'Regression': 0,
    'Classification': 1,
}

var ELM = function(d,L,m,alpha,mf,type) 
{
    this.d;       // the dimension of input layer
    this.L;       // the number of hidden nodes
    this.m;       // the dimension of output layer
    this.a;       // weight mat (input -> hidden)
    this.b;       // bias vec (input -> hidden)
    this.beta;    // weight mat (hidden -> output)
    this.alpha;   // L2 regularization coefficient
    this.mapfunc; // mapping function of hidden layer
    this.elmtype; // regression / classification

    this.d = d;
    this.L = L;
    this.m = m;
    this.alpha = alpha;

    this.a = numeric.sub(0.5, numeric.random([L,d]));
    this.b = numeric.sub(0.5, numeric.random([L]));

    this.mapfunc = mf;
    this.elmtype = type;
}

ELM.prototype.orthogonalize = function(x)
{
    var r = numeric.svd(x);
    return numeric.dot(r.U, numeric.transpose(r.V));
}

ELM.prototype.preprocess = function(T)
{
    if (this.elmtype == ELMType.Classification)
    {
        var N = numeric.dim(T)[0];
        var rT = numeric.zeros([N, this.m]);
        for (var i=0;i<N;i++) {
            rT[i][T[i][0]] = 1.0;
        }
        return rT;
    }
    else{
        return T;
    }
}

ELM.prototype.postprocess = function(Y)
{
    if (this.elmtype == ELMType.Classification)
    {
        var N = numeric.dim(Y)[0];
        var rY = numeric.zeros([N, 1]);
        for (var i=0;i<N;i++) {
            rY[i][0] = Y[i].indexOf(Math.max.apply(Math, Y[i]));
        }
        return rY;
    }
    else{
        return Y;
    }
}

ELM.prototype.train = function(X,T)
{
    T = this.preprocess(T);
    var Xdim = numeric.dim(X);
    var Tdim = numeric.dim(T);

    if (Xdim[0] != Tdim[0]) throw new Error("dimension mismatch");
    if (Xdim[1] != this.d) throw new Error("dimension mismatch");
    if (Tdim[1] != this.m) throw new Error("dimension mismatch");

    var N = Xdim[0];
    if (N > this.L)
    {
        var A = numeric.random([this.L, this.L]);
        var B = numeric.random([this.L, this.m]);
        var tT = numeric.transpose(T);
        var Hi = numeric.random([N]);
        var Hj = numeric.random([N]);
        for (var j=0; j < this.L; j++) {
            for (var k=0; k < N; k++) {
                Hj[k] = this.mapfunc(this.a[j], this.b[j], X[k]);
            }
            for (var i=j; i < this.L; i++) {
                for (var k=0; k < N; k++) {
                    Hi[k] = this.mapfunc(this.a[i], this.b[i], X[k]);
                }
                A[j][i] = numeric.dot(Hi,Hj);
                A[i][j] = A[j][i];
            }
            for (var i=0; i < this.m; i++) {
                B[j][i] = numeric.dot(tT[i],Hj);
            }
        }
        var ii = numeric.mul(this.alpha, numeric.identity(this.L));
        this.beta = numeric.dot(numeric.inv(numeric.add(A,ii)),B);
    }
    else{
        var H = numeric.random([N, this.L]);
        for (var j=0; j < N; j++) {
            for (var i=0; i < this.L; i++) {
                H[j][i] = this.mapfunc(this.a[i], this.b[i], X[j]);
            }
        }
        var tH = numeric.transpose(H);
        var ii = numeric.mul(this.alpha, numeric.identity(N));
        this.beta = numeric.dot(numeric.dot(tH, numeric.inv(numeric.add(numeric.dot(H,tH),ii))), T);
    }
}

//ELM.prototype.train = function(X,T)
//{
//    T = this.preprocess(T);
//    var Xdim = numeric.dim(X);
//    var Tdim = numeric.dim(T);
//
//    if (Xdim[0] != Tdim[0]) throw new Error("dimension mismatch");
//    if (Xdim[1] != this.d) throw new Error("dimension mismatch");
//    if (Tdim[1] != this.m) throw new Error("dimension mismatch");
//
//    var N = Xdim[0];
//    var H = numeric.random([N, this.L]);
//    for (var j=0; j < N; j++) {
//        for (var i=0; i < this.L; i++) {
//            H[j][i] = this.mapfunc(this.a[i], this.b[i], X[j]);
//        }
//    }
//
//    var tH = numeric.transpose(H);
//    if (N > this.L)
//    {
//        var ii = numeric.mul(this.alpha, numeric.identity(this.L));
//        this.beta = numeric.dot(numeric.dot(numeric.inv(numeric.add(numeric.dot(tH,H),ii)),tH), T);
//    }
//    else{
//        var ii = numeric.mul(this.alpha, numeric.identity(N));
//        this.beta = numeric.dot(numeric.dot(tH, numeric.inv(numeric.add(numeric.dot(H,tH),ii))), T);
//    }
//}

ELM.prototype.predict = function(X)
{
    var N = numeric.dim(X)[0];
    var H = numeric.random([N, this.L]);
    for (var j=0; j < N; j++) {
        for (var i=0; i < this.L; i++) {
            H[j][i] = this.mapfunc(this.a[i], this.b[i], X[j]);
        }
    }
    return this.postprocess(numeric.dot(H,this.beta));
}


var MappingFuncs = function() {};

MappingFuncs.Sigmoid = function(a,b,x){
    return 1.0/(1.0+Math.exp(-numeric.dot(a,x)+b));
}

MappingFuncs.HyperbolicTangent = function(a,b,x){
    var v = Math.exp(-numeric.dot(a,x)+b);
    return (1.0-v)/(1.0+v);
}

MappingFuncs.Gaussian = function(a,b,x){
    return Math.exp(-b*numeric.norm2(numeric.sub(x,a)));
}

MappingFuncs.Multiquadric = function(a,b,x){
    return Math.sqrt(numeric.norm2(numeric.add(b*b,numeric.sub(x,a))));
}

MappingFuncs.HardLimit = function(a,b,x){
    return numeric.dot(a,x)+b <= 0.0 ? 1.0 : 0.0;
}

MappingFuncs.Cosine = function(a,b,x){
    return Math.cos(numeric.dot(a,x)+b);
}

