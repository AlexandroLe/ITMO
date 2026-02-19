package org.example.domain;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

class DomainTest {

    @Test
    void boardAndDisembarkMovesPassengersAndTurnsLightOn() {
        Location outside = new Location("outside", "near car");
        Location reception = new Location("reception", "glass tables");
        Light light = new Light();
        Door door = new Door("front", outside, reception, light);

        Person arthur = new Person("Arthur", outside);
        Person oldMag = new Person("OldMag", outside);
        Aeromobile aero = new Aeromobile("a1");

        // preconditions
        assertTrue(outside.getOccupants().contains(arthur));
        assertTrue(outside.getOccupants().contains(oldMag));
        assertFalse(light.isOn());
        assertTrue(aero.getPassengers().isEmpty());

        // scenario
        aero.boardAllWithMarkers(Arrays.asList(arthur, oldMag));
        aero.approachDoor(door);
        aero.disembarkAllWithMarkers(reception);

        // postconditions
        assertTrue(light.isOn(), "Light should be on after approach");
        assertTrue(reception.getOccupants().contains(arthur), "Arthur must be in reception");
        assertTrue(reception.getOccupants().contains(oldMag), "OldMag must be in reception");
        assertTrue(aero.getPassengers().isEmpty(), "Aeromobile should be empty after disembark");
        assertSame(reception, arthur.getLocation());
    }

    @Test
    void lockedDoorPreventsPassageAndApproachTurnsLightOn() {
        Location outside = new Location("outside", "");
        Location reception = new Location("reception", "");
        Light light = new Light();
        Door door = new Door("front", outside, reception, light);
        door.lock();

        Person p = new Person("P", outside);

        // approach should still turn on light
        door.approachWithMarkers();
        assertTrue(light.isOn(), "Light must be on after approach even if the door is locked");

        // passThrough must throw
        IllegalStateException ex = assertThrows(IllegalStateException.class, () -> door.passThrough(outside));
        assertTrue(ex.getMessage().toLowerCase().contains("lock"));
    }

    @Test
    void personMoveToUpdatesOccupantsLists() {
        Location car = new Location("car", "in car");
        Location reception = new Location("reception", "room");
        Person p = new Person("Testy", car);

        assertTrue(car.getOccupants().contains(p));
        assertSame(car, p.getLocation());

        p.moveTo(reception);

        assertSame(reception, p.getLocation());
        assertFalse(car.getOccupants().contains(p));
        assertTrue(reception.getOccupants().contains(p));
    }

    @Test
    void receptionContainsItemsAndDescription() {
        Location reception = new Location("reception", "glass tables and trophies");
        Item table = new Item("glass table", "glass");
        Item trophy = new Item("trophy", "plastic");

        reception.addItem(table);
        reception.addItem(trophy);

        assertEquals(2, reception.getItems().size());
        assertTrue(reception.getItems().contains(table));
        assertTrue(reception.getItems().contains(trophy));
        assertEquals("glass tables and trophies", reception.getDescription());
    }

    @Test
    void repeatedBoardingDoesNotDuplicatePassengers() {
        Location outside = new Location("outside", "");
        Person p = new Person("Solo", outside);
        Aeromobile aero = new Aeromobile("aX");

        aero.boardAllWithMarkers(Arrays.asList(p));
        aero.boardAllWithMarkers(Arrays.asList(p)); // second time

        assertEquals(1, aero.getPassengers().size(), "Passenger should not be duplicated in vehicle");
    }

    // --- Additional tests ---

    @Test
    void doorOtherSideIsSymmetric() {
        Location l1 = new Location("L1", "");
        Location l2 = new Location("L2", "");
        Door d = new Door("d", l1, l2, new Light());

        assertSame(l2, d.otherSide(l1));
        assertSame(l1, d.otherSide(l2));
        assertNull(d.otherSide(new Location("other", "")));
    }

    @Test
    void moveToNullThrows() {
        Person p = new Person("p");
        assertThrows(IllegalArgumentException.class, () -> p.moveTo(null));
    }

    @Test
    void disembarkOnEmptyVehicleIsNoOp() {
        Location dest = new Location("dest", "");
        Aeromobile empty = new Aeromobile("empty");
        // should not throw
        empty.disembarkAllWithMarkers(dest);
        // and dest remains empty
        assertTrue(dest.getOccupants().isEmpty());
    }

    @Test
    void addSameItemDoesNotDuplicate() {
        Location reception = new Location("reception", "");
        Item trophy = new Item("trophy", "plastic");
        reception.addItem(trophy);
        reception.addItem(trophy); // second time
        assertEquals(1, reception.getItems().size());
    }

    @Test
    void movingToSameLocationDoesNotDuplicateOccupant() {
        Location room = new Location("room", "");
        Person p = new Person("X", room);
        // move to same location
        p.moveTo(room);
        // only one occupant
        assertEquals(1, room.getOccupants().size());
    }

    @Test
    void approachIsIdempotentForLight() {
        Location outside = new Location("outside", "");
        Location reception = new Location("reception", "");
        Light light = new Light();
        Door door = new Door("d", outside, reception, light);
        // call approach twice
        door.approachWithMarkers();
        door.approachWithMarkers();
        assertTrue(light.isOn(), "light should remain on after repeated approach");
    }

    @Test
    void boardingRemovesFromOldLocation() {
        Location outside = new Location("outside", "");
        Location other = new Location("other", "");
        Person p = new Person("Z", other);
        Aeromobile aero = new Aeromobile("v");
        assertTrue(other.getOccupants().contains(p));
        aero.boardAllWithMarkers(Arrays.asList(p));
        assertFalse(other.getOccupants().contains(p), "boarding should remove person from previous location");
        assertTrue(aero.getPassengers().contains(p));
    }
}
