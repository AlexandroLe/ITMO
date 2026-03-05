package org.example.domain;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LightTest {

    private Light light;

    @BeforeEach
    void setUp() {
        light = new Light();
    }

    @Test
    void constructor_ShouldSetInitialStateToOff() {
        assertFalse(light.isOn());
    }

    @Test
    void turnOn_ShouldSetStateToTrue() {
        light.turnOn();
        assertTrue(light.isOn());
    }

    @Test
    void turnOn_WhenAlreadyOn_ShouldStayTrue() {
        light.turnOn();
        light.turnOn();
        assertTrue(light.isOn());
    }

    @Test
    void turnOff_ShouldSetStateToFalse() {
        light.turnOn();
        light.turnOff();
        assertFalse(light.isOn());
    }

    @Test
    void turnOff_WhenAlreadyOff_ShouldStayFalse() {
        light.turnOff();
        assertFalse(light.isOn());
        light.turnOff();
        assertFalse(light.isOn());
    }

    @Test
    void isOn_ShouldReflectCurrentState() {
        assertFalse(light.isOn());
        light.turnOn();
        assertTrue(light.isOn());
        light.turnOff();
        assertFalse(light.isOn());
    }
}