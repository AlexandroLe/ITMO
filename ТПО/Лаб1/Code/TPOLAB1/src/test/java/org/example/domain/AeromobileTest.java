package org.example.domain;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class AeromobileTest {

    private Aeromobile aeromobile;
    private Person arthur;
    private Person oldMag;
    private Location outside;
    private Location reception;
    private Door frontDoor;
    private Light light;

    @BeforeEach
    void setUp() {
        aeromobile = new Aeromobile("aero-1");
        outside = new Location("outside", "рядом с дверью");
        reception = new Location("reception", "приемная");
        light = new Light();
        frontDoor = new Door("frontDoor", outside, reception, light);
        arthur = new Person("Arthur", outside);
        oldMag = new Person("OldMag", outside);
    }

    @Test
    void constructor_ShouldInitializeWithEmptyPassengers() {
        assertTrue(aeromobile.getPassengers().isEmpty());
    }

    @Test
    void getPassengers_ShouldReturnUnmodifiableList() {
        List<Person> passengers = aeromobile.getPassengers();
        assertThrows(UnsupportedOperationException.class, () -> passengers.add(arthur));
    }

    @Test
    void boardAllWithMarkers_ShouldAddNewPassengers() {
        aeromobile.boardAllWithMarkers(Arrays.asList(arthur, oldMag));

        assertEquals(2, aeromobile.getPassengers().size());
        assertTrue(aeromobile.getPassengers().contains(arthur));
        assertTrue(aeromobile.getPassengers().contains(oldMag));
    }

    @Test
    void boardAllWithMarkers_ShouldRemovePassengersFromPreviousLocation() {
        assertTrue(outside.getOccupants().contains(arthur));
        assertTrue(outside.getOccupants().contains(oldMag));

        aeromobile.boardAllWithMarkers(Arrays.asList(arthur, oldMag));

        assertFalse(outside.getOccupants().contains(arthur));
        assertFalse(outside.getOccupants().contains(oldMag));
    }

    @Test
    void boardAllWithMarkers_ShouldNotChangePersonsLocationField() {
        aeromobile.boardAllWithMarkers(List.of(arthur));

        assertEquals(outside, arthur.getLocation());
        assertFalse(outside.getOccupants().contains(arthur));
    }

    @Test
    void boardAllWithMarkers_ShouldIgnoreDuplicatePassengers() {
        aeromobile.boardAllWithMarkers(List.of(arthur));
        aeromobile.boardAllWithMarkers(List.of(arthur));

        assertEquals(1, aeromobile.getPassengers().size());
        assertTrue(aeromobile.getPassengers().contains(arthur));
    }

    @Test
    void boardAllWithMarkers_ShouldHandlePassengerWithNullLocation() {
        Person nobody = new Person("Nobody");
        aeromobile.boardAllWithMarkers(List.of(nobody));

        assertEquals(1, aeromobile.getPassengers().size());
        assertTrue(aeromobile.getPassengers().contains(nobody));
        assertNull(nobody.getLocation());
    }

    @Test
    void boardAllWithMarkers_WithEmptyList_ShouldDoNothing() {
        aeromobile.boardAllWithMarkers(List.of());
        assertTrue(aeromobile.getPassengers().isEmpty());
    }

    @Test
    void approachDoor_ShouldTurnOnDoorLight() {
        assertFalse(light.isOn());

        aeromobile.approachDoor(frontDoor);

        assertTrue(light.isOn());
    }

    @Test
    void approachDoor_ShouldWorkEvenIfDoorHasNoLight() {
        Door doorWithoutLight = new Door("noLight", outside, reception, null);
        aeromobile.approachDoor(doorWithoutLight);
    }

    @Test
    void disembarkAllWithMarkers_ShouldMoveAllPassengersToDestination() {
        aeromobile.boardAllWithMarkers(Arrays.asList(arthur, oldMag));
        assertFalse(reception.getOccupants().contains(arthur));
        assertFalse(reception.getOccupants().contains(oldMag));

        aeromobile.disembarkAllWithMarkers(reception);

        assertTrue(reception.getOccupants().contains(arthur));
        assertTrue(reception.getOccupants().contains(oldMag));
        assertEquals(reception, arthur.getLocation());
        assertEquals(reception, oldMag.getLocation());
    }

    @Test
    void disembarkAllWithMarkers_ShouldClearPassengerList() {
        aeromobile.boardAllWithMarkers(Arrays.asList(arthur, oldMag));
        aeromobile.disembarkAllWithMarkers(reception);

        assertTrue(aeromobile.getPassengers().isEmpty());
    }

    @Test
    void disembarkAllWithMarkers_WithEmptyPassengers_ShouldDoNothing() {
        aeromobile.disembarkAllWithMarkers(reception);
        assertTrue(reception.getOccupants().isEmpty());
        assertTrue(aeromobile.getPassengers().isEmpty());
    }

    @Test
    void disembarkAllWithMarkers_ShouldHandlePassengerWithInconsistentLocation() {
        aeromobile.boardAllWithMarkers(List.of(arthur));

        aeromobile.disembarkAllWithMarkers(reception);

        assertEquals(reception, arthur.getLocation());
        assertTrue(reception.getOccupants().contains(arthur));
        assertFalse(outside.getOccupants().contains(arthur));
    }
}