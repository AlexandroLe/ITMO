package org.example.domain;

public class Item {
    private final String name;
    private final String material;

    public Item(String name, String material) {
        this.name = name;
        this.material = material;
    }

    public String getName() { return name; }
    public String getMaterial() { return material; }

    @Override
    public String toString() { return "Item{" + name + ":" + material + "}"; }
}
