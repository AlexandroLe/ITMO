package org.example.domain;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        Location outside = new Location("outside", "рядом с дверью и машиной");
        Location reception = new Location("reception", "приемная уставленная стеклянными столиками и наградами");
        Light doorLight = new Light();

        Door frontDoor = new Door("frontDoor", outside, reception, doorLight);

        Person arthur = new Person("Arthur", outside);
        Person oldMag = new Person("OldMag", outside);

        Aeromobile aero = new Aeromobile("aero-1");

        aero.boardAllWithMarkers(Arrays.asList(arthur, oldMag));
        aero.approachDoor(frontDoor);
        aero.disembarkAllWithMarkers(reception);
        if (!reception.getOccupants().isEmpty()) {
            reception.getOccupants().get(0).inspectWithMarkers();
        }

        System.out.println("\n--- final state ---");
        System.out.println("light on: " + frontDoor.getLight().isOn());
        System.out.println("reception occupants: " + reception.getOccupants());
        System.out.println("aeromobile passengers: " + aero.getPassengers());
    }
}
