package org.example.domain;

public class Person {
    private final String name;
    private Location location;

    public Person(String name) { this.name = name; }

    public Person(String name, Location initial) {
        this.name = name;
        if (initial != null) {
            this.location = initial;
            initial.addOccupant(this);
        }
    }

    public String getName() { return name; }
    public Location getLocation() { return location; }

    public void moveTo(Location dest) {
        if (dest == null) throw new IllegalArgumentException("dest null");
        if (location != null) location.removeOccupant(this);
        location = dest;
        dest.addOccupant(this);
        System.out.println(name + " moved to " + dest.getName());
    }

    public void inspectWithMarkers() {
        System.out.print(" |D| ");
        System.out.println(name + " inspects " + (location == null ? "nowhere" : location.getName()));
        System.out.print(" |d| ");
    }

    @Override
    public String toString() {
        return "Person{" + name + "}";
    }
}
