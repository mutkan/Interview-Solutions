// coding
// design
// algorithm
// communication
// efficiency


List<Pair<Integer, Integer>> points

[a, b, c, d, e]

Dab == Dac

ABC, ACD, ADE, ABE, ABD, ACE 1+2+3+..+99

AB, AC, AD, AE…
10, 11, 10, 10
10 -3, 11 -1, 2+1


Map<Point, Map<distance, count>>

Class Solution {
	public static int findIsoscelesTriangles(List<Pair<Integer, Integer>> points) {
		
		for (int i = 0; i < list.size(); i++) {
			//A
			for (int j = 0; j < list.size(); j++) {
				HashMap<Integer, Integer> distanceMap = new HashMap<>();
				int distance = Math.abs(list(i).first - list(j).first) + Math.abs(list(i).second - list(j).second);
				If (distanceMap.contains(distance)) {
					distanceMap.put(distance, distanceMap.get(distance) + 1);
} else {
				distanceMap.put(distance, 1);
                        }
			}
			for (Integer value: distanceMap.values()) {

} 
}
	}
}
