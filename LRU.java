private Node head;
private Node end;
private int limit;

private HashMap<String, Node> hashMap;

public LRUCache(int limit) {
	this.limit = limit;
	hashMap = new HashMap<String, Node>();
}

public String get(String key) {
	Node node = hashMap.get(key);
	if(node == null) return null;
	refreshNode(node);
	return node.value;
}

public void put(String key, String value) {
	Node node = hashMap.get(key);
	if(node == null) {
		if(hashMap.size() >= limit) {
			String oldKey = removeNode(head);
			hashMap.remove(oldKey);
		}
		node = new Node(key, value);
		addNode(node);
		hashMap.put(key, node);
	} else {
		node.value = value;
		refreshNode(node);
	}
}

public void remove(String key) {
	Node node = hashMap.get(key);
	removeNode(node);
	hashMap.remove(key);
}

private void refreshNode(Node node) {
	if(node == end) return;
	removeNode(node);
	addNode(node);
}

private String removeNode(Node node) {
	if(node == end) {
		end = end.pre;
	} else if(node == head) {
		head = head.next;
	} else {
		node.pre.next = node.next;
		node.next.pre = node.pre;
	}
	return node.key;
}

private void addNode(Node node) {
	if(end != null) {
		end.next = node;
		node.pre = end;
		node.next = null;
	}
	end = node;
	if(head == null) {
		head = node;
	}
}

class Node {
	Node(String key, String value) {
		this.key = key;
		this.value = value;
	}

	public Node pre;
	public Node next;
	public String key;
	public String value;
}

//test
public static void main(String[] args) {
	LRUCache lruCache = new LRUCache(5);
	lruCache.put("001", "user_info");
	lruCache.put("002", "user_info");
	lruCache.put("003", "user_info");
	lruCache.put("004", "user_info");
	lruCache.put("005", "user_info");
	lruCache.get("002");
	lruCache.put("004", "user_update");
	lruCache.put("006", "user_info");
	System.out.println(lruCache.get("001"));
	System.out.println(lruCache.get("006"));
}

//go check database Redis