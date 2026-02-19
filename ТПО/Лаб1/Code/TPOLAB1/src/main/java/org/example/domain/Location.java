package org.example.domain;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Location {
    private final String name;
    private final String description;
    private final List<Person> occupants = new ArrayList<>();
    private final List<Item> items = new ArrayList<>();
    private final List<Door> doors = new ArrayList<>();

    public Location(String name, String description) {
        this.name = name;
        this.description = description;
    }

    public String getName() { return name; }
    public String getDescription() { return description; }

    void addOccupant(Person p) {
        if (!occupants.contains(p)) occupants.add(p);
    }

    void removeOccupant(Person p) {
        occupants.remove(p);
    }

    public List<Person> getOccupants() { return Collections.unmodifiableList(occupants); }
    public List<Item> getItems() { return Collections.unmodifiableList(items); }
    public List<Door> getDoors() { return Collections.unmodifiableList(doors); }

    public void addItem(Item it) { if (it != null && !items.contains(it)) items.add(it); }
    public void addDoor(Door d) { if (d != null && !doors.contains(d)) doors.add(d); }

    @Override
    public String toString() {
        return "Location{" + name + "}";
    }
}
