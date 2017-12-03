package me.peyppicp

import org.nd4j.linalg.factory.Nd4j


fun main(args: Array<String>) {
    // TODO Auto-generated method stub
    // 创建一个0元素的矩阵 2*2*2
//    val m1 = Nd4j.create(*intArrayOf(2, 2, 2))
//    println(m1)
//    // 创建一个随机元素的矩阵 2*2*2
//    val m2 = Nd4j.rand(intArrayOf(2, 2, 2))
//    println(m2)
//    // 矩阵的加法
//    println(m1.add(m2))
//    // 矩阵的减法
//    println(m1.sub(m2))
//    // 矩阵的乘法
//    println(m1.mul(m2))
//    // 矩阵的展开
//    val m3 = Nd4j.toFlattened(m1)
//    println(m3)
//    // 矩阵的reshape
//    println(m3.reshape(2, 2, 2))
//    // 矩阵中添加元素 在 [0,?,0]位置 第二个维度插入
//    m1.put(arrayOf(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(0)), Nd4j.create(doubleArrayOf(0.1, 0.2)))
//    println(m1)
//    // 具体的某一个元素
//    m1.putScalar(intArrayOf(0, 0, 0), 12.0)
//    println(m1)

    val create = Nd4j.create(intArrayOf(2, 2, 12), 'f')
    println(create)
    create.putScalar(intArrayOf(0,1,0),1)
    println(create)
}