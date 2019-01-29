// you can write to stdout for debugging purposes, e.g.
// console.log('this is a debug message');

function solution(ranks) {
    // write your code in JavaScript (Node.js 8.9.4)
    var sum = 0;
    var soldiers = new Map();
    for (var i = 0; i < ranks.length; i++) {
        if (soldiers.has(ranks[i])) {
            var temp = soldiers.get(ranks[i]);
            soldiers.delete(ranks[i]);
            soldiers.set(ranks[i], temp + 1);
        } else {
            soldiers.set(ranks[i], 1);
        }
    }
    for (const k of soldiers.keys()) {
        if (soldiers.has(k+1)) {
            sum += soldiers.get(k);
        }
    }
    return sum;
}