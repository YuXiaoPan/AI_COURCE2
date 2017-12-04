package me.peyppicp.advance;

import com.google.common.base.Joiner;
import com.vdurmont.emoji.Emoji;
import com.vdurmont.emoji.EmojiManager;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * @author YuXiao Pan
 * @date 2017/12/4
 * @email yuxiao.pan@kikatech.com
 */
public class FrequencyStatisticsMain {

    public static void main(String[] args) throws IOException {
        File file = new File("commonLabelWithIndex.txt");
        List<String> lines = FileUtils.readLines(file, Charsets.UTF_8);
        HashMap<String, Integer> map = new HashMap<>();
        for (String line : lines) {
            for (Emoji emoji : EmojiManager.getAll()) {
                if (line.contains(emoji.getUnicode())) {
                    map.merge(emoji.getUnicode(), 1, (a, b) -> a + b);
                }
            }
        }
        LinkedHashMap<String, Integer> tempMap = new LinkedHashMap<>();
        map.entrySet().parallelStream().sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .forEachOrdered(e -> tempMap.put(e.getKey(), e.getValue()));
        System.out.println(Joiner.on("\n").skipNulls().join(tempMap.entrySet()));
    }
}
