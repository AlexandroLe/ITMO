package org.example.domain;

public class Door {
    private final String name;
    private final Location a;
    private final Location b;
    private final Light light;
    private boolean locked = false;

    public Door(String name, Location a, Location b, Light light) {
        this.name = name;
        this.a = a;
        this.b = b;
        this.light = light;
        a.addDoor(this);
        b.addDoor(this);
    }

    public String getName() { return name; }
    public Light getLight() { return light; }

    public void lock() { locked = true; }
    public void unlock() { locked = false; }
    public boolean isLocked() { return locked; }

    public Location otherSide(Location from) {
        if (from == null) return null;
        if (from.equals(a)) return b;
        if (from.equals(b)) return a;
        return null;
    }

    public void approachWithMarkers() {
        System.out.print(" |B| ");
        if (light != null) light.turnOn();
        System.out.print(" |b| ");
    }

    public Location passThrough(Location from) {
        if (locked) throw new IllegalStateException("Door is locked");
        return otherSide(from);
    }

    @Override
    public String toString() {
        return "Door{" + name + "}";
    }
}
