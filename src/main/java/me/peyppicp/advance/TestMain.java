package me.peyppicp.advance;

import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Optional;

public class TestMain {

    public static void main(String[] args) throws IOException {
        File file = new File("F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\distinctLines.txt");
        List<String> strings = FileUtils.readLines(file, Charsets.UTF_8);
        File file1 = new File("F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\standardData.txt");
        List<String> strings1 = FileUtils.readLines(file1, Charsets.UTF_8);
        String temp = "fuck both of";
        Optional<String> first = strings1.parallelStream().filter(s -> s.contains(temp)).findFirst();
        System.out.println(first.get());
    }
}
