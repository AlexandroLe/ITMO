package org.example.domain;

public class Light {
    private boolean on = false;
    public boolean isOn() { return on; }
    public void turnOn() { on = true; System.out.println("light:on"); }
    public void turnOff() { on = false; System.out.println("light:off"); }
}
