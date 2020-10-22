
public class Event {
    final private String type;
    final private double value;
    final private int count;

    public Event(String type, double value, int count) {
        this.type = type;
        this.value = value;
        this.count = count;
    }

    public String getType() {
        return type;
    }
    public double getValue() {
        return value;
    }
    public int getCount() { return count; }

}

