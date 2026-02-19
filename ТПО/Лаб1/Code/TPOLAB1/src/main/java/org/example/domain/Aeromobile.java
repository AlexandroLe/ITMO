package org.example.domain;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Aeromobile {
    private final String id;
    private final List<Person> passengers = new ArrayList<>();

    public Aeromobile(String id) { this.id = id; }

    public List<Person> getPassengers() { return Collections.unmodifiableList(passengers); }

    public void boardAllWithMarkers(List<Person> list) {
        System.out.print(" |A| ");
        for (Person p : list) {
            if (!passengers.contains(p)) {
                if (p.getLocation() != null) p.getLocation().removeOccupant(p);
                passengers.add(p);
                System.out.println("boarded " + p.getName());
            }
        }
        System.out.print(" |a| ");
    }

    public void approachDoor(Door door) {
        door.approachWithMarkers();
    }

    public void disembarkAllWithMarkers(Location dest) {
        System.out.print(" |C| ");
        for (Person p : new ArrayList<>(passengers)) {
            p.moveTo(dest);
            passengers.remove(p);
        }
        System.out.print(" |c| ");
    }

    @Override
    public String toString() { return "Aeromobile{" + id + "}"; }
}
