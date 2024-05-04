Mac M1 use  ARM assembly language,

X86 machine use x86 assembly language



Want to take a look at the libc lib?

read these two libs.


```
git clone https://github.com/bminor/glibc.git
```

```
git clone https://github.com/kraj/musl.git
```


# reference


https://securityboulevard.com/2021/05/linux-x86-assembly-how-to-build-a-hello-world-program-in-nasm/

https://smist08.wordpress.com/2021/01/08/apple-m1-assembly-language-hello-world/

https://blog.csdn.net/techforward/article/details/138390530?spm=1001.2014.3001.5501


In glibc repo, you could observe that **puts**  represents **`_IO_puts`**



`weak_alias (_IO_puts, puts) `
