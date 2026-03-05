package org.example.domain;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DoorTest {

    private Location locA;
    private Location locB;
    private Light light;
    private Door door;

    @BeforeEach
    void setUp() {
        locA = new Location("A", "Location A");
        locB = new Location("B", "Location B");
        light = new Light();
        door = new Door("testDoor", locA, locB, light);
    }

    @Test
    void constructor_ShouldInitializeFieldsAndAddItselfToLocations() {
        assertEquals("testDoor", door.getName());
        assertSame(light, door.getLight());
        assertFalse(door.isLocked());

        assertTrue(locA.getDoors().contains(door));
        assertTrue(locB.getDoors().contains(door));
    }

    @Test
    void constructor_ShouldAllowNullLight() {
        Door doorNoLight = new Door("noLight", locA, locB, null);
        assertNull(doorNoLight.getLight());
        assertTrue(locA.getDoors().contains(doorNoLight));
        assertTrue(locB.getDoors().contains(doorNoLight));
    }

    @Test
    void getName_ShouldReturnCorrectName() {
        assertEquals("testDoor", door.getName());
    }

    @Test
    void getLight_ShouldReturnLightInstance() {
        assertSame(light, door.getLight());
    }

    @Test
    void lock_ShouldSetLockedToTrue() {
        door.lock();
        assertTrue(door.isLocked());
    }

    @Test
    void unlock_ShouldSetLockedToFalse() {
        door.lock(); // сначала закрываем
        door.unlock();
        assertFalse(door.isLocked());
    }

    @Test
    void isLocked_ShouldReturnFalseByDefault() {
        assertFalse(door.isLocked());
    }

    @Test
    void otherSide_WhenFromIsA_ShouldReturnB() {
        assertEquals(locB, door.otherSide(locA));
    }

    @Test
    void otherSide_WhenFromIsB_ShouldReturnA() {
        assertEquals(locA, door.otherSide(locB));
    }

    @Test
    void otherSide_WhenFromIsNeither_ShouldReturnNull() {
        Location other = new Location("C", "Other");
        assertNull(door.otherSide(other));
    }

    @Test
    void otherSide_WhenFromIsNull_ShouldReturnNull() {
        assertNull(door.otherSide(null));
    }

    @Test
    void approachWithMarkers_ShouldTurnOnLightIfPresent() {
        assertFalse(light.isOn());
        door.approachWithMarkers();
        assertTrue(light.isOn());
    }

    @Test
    void approachWithMarkers_WhenLightIsNull_ShouldNotThrow() {
        Door doorNoLight = new Door("noLight", locA, locB, null);
        assertDoesNotThrow(doorNoLight::approachWithMarkers);
    }

    @Test
    void passThrough_WhenDoorUnlocked_ShouldReturnOtherSide() {
        assertEquals(locB, door.passThrough(locA));
        assertEquals(locA, door.passThrough(locB));
    }

    @Test
    void passThrough_WhenDoorLocked_ShouldThrowIllegalStateException() {
        door.lock();
        IllegalStateException exception = assertThrows(IllegalStateException.class,
                () -> door.passThrough(locA));
        assertEquals("Door is locked", exception.getMessage());
    }

    @Test
    void passThrough_ShouldDelegateToOtherSide() {
        Location unknown = new Location("C", "Unknown");
        assertNull(door.passThrough(unknown));
    }
}