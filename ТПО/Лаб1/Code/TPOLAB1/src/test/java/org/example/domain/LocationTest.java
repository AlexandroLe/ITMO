package org.example.domain;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class LocationTest {

    private Location location;
    private Person person1;
    private Person person2;
    private Item item1;
    private Item item2;
    private Door door1;
    private Door door2;
    private Location otherLocation;

    @BeforeEach
    void setUp() {
        location = new Location("testLoc", "Test description");
        otherLocation = new Location("other", "Other location");
        person1 = new Person("Alice");
        person2 = new Person("Bob");
        item1 = new Item("book", "paper");
        item2 = new Item("key", "metal");
        Light light = new Light();
        door1 = new Door("door1", location, otherLocation, light);
        door2 = new Door("door2", otherLocation, location, light);
    }

    @Test
    void constructor_ShouldSetNameAndDescription() {
        assertEquals("testLoc", location.getName());
        assertEquals("Test description", location.getDescription());
    }


    @Test
    void addOccupant_ShouldAddPersonToOccupants() {
        location.addOccupant(person1);
        List<Person> occupants = location.getOccupants();
        assertEquals(1, occupants.size());
        assertTrue(occupants.contains(person1));
    }

    @Test
    void addOccupant_ShouldNotAddDuplicatePerson() {
        location.addOccupant(person1);
        location.addOccupant(person1);
        assertEquals(1, location.getOccupants().size());
    }

    @Test
    void removeOccupant_ShouldRemovePersonFromOccupants() {
        location.addOccupant(person1);
        location.addOccupant(person2);
        location.removeOccupant(person1);
        List<Person> occupants = location.getOccupants();
        assertEquals(1, occupants.size());
        assertFalse(occupants.contains(person1));
        assertTrue(occupants.contains(person2));
    }

    @Test
    void removeOccupant_WhenPersonNotPresent_ShouldDoNothing() {
        location.addOccupant(person1);
        location.removeOccupant(person2);
        assertEquals(1, location.getOccupants().size());
        assertTrue(location.getOccupants().contains(person1));
    }

    @Test
    void getOccupants_ShouldReturnUnmodifiableList() {
        location.addOccupant(person1);
        List<Person> occupants = location.getOccupants();
        assertThrows(UnsupportedOperationException.class, () -> occupants.add(person2));
    }

    @Test
    void addItem_ShouldAddItem() {
        location.addItem(item1);
        List<Item> items = location.getItems();
        assertEquals(1, items.size());
        assertTrue(items.contains(item1));
    }

    @Test
    void addItem_ShouldNotAddDuplicateItem() {
        location.addItem(item1);
        location.addItem(item1);
        assertEquals(1, location.getItems().size());
    }

    @Test
    void addItem_WithNull_ShouldDoNothing() {
        location.addItem(null);
        assertTrue(location.getItems().isEmpty());
    }

    @Test
    void getItems_ShouldReturnUnmodifiableList() {
        location.addItem(item1);
        List<Item> items = location.getItems();
        assertThrows(UnsupportedOperationException.class, () -> items.add(item2));
    }

    @Test
    void addDoor_ShouldAddDoor() {
        Location newLoc = new Location("new", "new");
        Door newDoor = new Door("newDoor", newLoc, otherLocation, null);
        location.addDoor(newDoor);
        assertTrue(location.getDoors().contains(newDoor));
    }

    @Test
    void addDoor_ShouldNotAddDuplicateDoor() {
        int beforeSize = location.getDoors().size();
        location.addDoor(door1);
        assertEquals(beforeSize, location.getDoors().size());
    }

    @Test
    void addDoor_WithNull_ShouldDoNothing() {
        location.addDoor(null);
        assertNotNull(location.getDoors());
    }

    @Test
    void getDoors_ShouldReturnUnmodifiableList() {
        List<Door> doors = location.getDoors();
        assertThrows(UnsupportedOperationException.class, () -> doors.add(door2));
    }


    @Test
    void occupantsAndDoorsAreIndependent() {
        location.addOccupant(person1);
        assertTrue(location.getOccupants().contains(person1));
        assertFalse(location.getDoors().contains(person1));
    }
}