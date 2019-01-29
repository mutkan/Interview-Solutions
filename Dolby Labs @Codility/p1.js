// you can write to stdout for debugging purposes, e.g.
// console.log('this is a debug message');

function solution(A, B) {
    // write your code in JavaScript (Node.js 8.9.4)
    var stringA = A.toString();
    var stringB = B.toString();
    var a = stringA.length;
    var b = stringB.length;
    
    for (var i = 0; i<= b - a; i++) {
        var j;
        for (j = 0; j < a; j++) {
            if (stringB.charAt(i+j) != stringA.charAt(j)) {
                break;
            }
        }
        if (j == a) {
            return i;
        }
    }
    return -1;
}