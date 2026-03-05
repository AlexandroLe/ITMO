package org.example.domain;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PersonTest {

    private Location loc1;
    private Location loc2;
    private Person person;

    @BeforeEach
    void setUp() {
        loc1 = new Location("room1", "First room");
        loc2 = new Location("room2", "Second room");
        person = new Person("Alice", loc1);
    }

    @Test
    void constructorWithNameOnly_ShouldCreatePersonWithNullLocation() {
        Person p = new Person("Bob");
        assertEquals("Bob", p.getName());
        assertNull(p.getLocation());
    }

    @Test
    void constructorWithNameAndLocation_ShouldSetLocationAndAddToOccupants() {
        Person p = new Person("Charlie", loc1);
        assertEquals("Charlie", p.getName());
        assertSame(loc1, p.getLocation());
        assertTrue(loc1.getOccupants().contains(p));
    }

    @Test
    void constructorWithNameAndNullLocation_ShouldNotAddToAnyLocation() {
        Person p = new Person("David", null);
        assertEquals("David", p.getName());
        assertNull(p.getLocation());
        assertFalse(loc1.getOccupants().contains(p));
        assertFalse(loc2.getOccupants().contains(p));
    }

    @Test
    void getLocation_ShouldReturnCurrentLocation() {
        assertSame(loc1, person.getLocation());
    }

    @Test
    void moveTo_ShouldChangeLocationAndUpdateOccupants() {
        person.moveTo(loc2);
        assertSame(loc2, person.getLocation());
        assertTrue(loc2.getOccupants().contains(person));
        assertFalse(loc1.getOccupants().contains(person));
    }

    @Test
    void moveTo_WhenAlreadyAtDestination_ShouldStillBeConsistent() {
        person.moveTo(loc1);
        assertSame(loc1, person.getLocation());
        assertTrue(loc1.getOccupants().contains(person));
        assertEquals(1, loc1.getOccupants().stream().filter(p -> p.equals(person)).count());
    }

    @Test
    void moveTo_WhenPersonHasNoLocation_ShouldSetLocationAndAddToOccupants() {
        Person p = new Person("Eve");
        p.moveTo(loc1);
        assertSame(loc1, p.getLocation());
        assertTrue(loc1.getOccupants().contains(p));
    }

    @Test
    void moveTo_WithNullDestination_ShouldThrowIllegalArgumentException() {
        assertThrows(IllegalArgumentException.class, () -> person.moveTo(null));
    }

    @Test
    void inspectWithMarkers_WhenLocationIsNotNull_ShouldNotThrow() {
        assertDoesNotThrow(() -> person.inspectWithMarkers());
    }

    @Test
    void inspectWithMarkers_WhenLocationIsNull_ShouldNotThrow() {
        Person p = new Person("Frank");
        assertDoesNotThrow(p::inspectWithMarkers);
    }

    @Test
    void moveTo_ShouldHandleMultipleMovesCorrectly() {
        Location loc3 = new Location("loc3", "third");
        person.moveTo(loc2);
        person.moveTo(loc3);
        assertFalse(loc1.getOccupants().contains(person));
        assertFalse(loc2.getOccupants().contains(person));
        assertTrue(loc3.getOccupants().contains(person));
        assertSame(loc3, person.getLocation());
    }

    @Test
    void moveTo_ShouldNotCreateDuplicatesOnSameLocation() {
        person.moveTo(loc2);
        person.moveTo(loc2);
        assertEquals(1, loc2.getOccupants().stream().filter(p -> p.equals(person)).count());
        assertFalse(loc1.getOccupants().contains(person));
    }
}