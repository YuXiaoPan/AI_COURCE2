package me.peyppicp

import com.vdurmont.emoji.EmojiManager
import java.io.File
import java.util.*

fun getMinimalCharacterSet(): CharArray {
    val validChars = LinkedList<Char>()
    run {
        var c = 'a'
        while (c <= 'z') {
            validChars.add(c)
            c++
        }
    }
    run {
        var c = 'A'
        while (c <= 'Z') {
            validChars.add(c)
            c++
        }
    }
    var c = '0'
    while (c <= '9') {
        validChars.add(c)
        c++
    }
    val temp = charArrayOf('!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t')
    for (ch in temp) validChars.add(ch)
    val out = CharArray(validChars.size)
    var i = 0
    for (ch in validChars) out[i++] = ch
    return out
}

fun getDefaultCharacterSet(): CharArray {
    val validChars = LinkedList<Char>()
    for (c in getMinimalCharacterSet()) validChars.add(c)
    val additionalChars = charArrayOf('@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_', '\\', '|', '<', '>')
    for (c in additionalChars) validChars.add(c)
    EmojiManager.getAll().forEach { it.unicode.toCharArray().forEach { validChars.add(it) } }
    val out = CharArray(validChars.size)
    var i = 0
    for (c in validChars) out[i++] = c
    return out
}

fun main(args: Array<String>) {
    val file = File("F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\standardData.txt")
    val readLines = file.readLines(Charsets.UTF_8)
    val defaultCharacterSet = getDefaultCharacterSet()
    for (lines in readLines) {
        for (cs in defaultCharacterSet) {
            for (oc in lines.toCharArray()) {
                if (oc == cs) {

                }
            }
        }
    }
    println()
}
