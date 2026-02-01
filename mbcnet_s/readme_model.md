

 `avg_pool = x.mean(dim=(2, 3))`
 ![avg](./uses_image/m_1.png)

 ` weights = self.sigmoid(self.relu(self.conv(pooled.unsqueeze(1))))`

![avg](./uses_image/m_2.png)
![avg](./uses_image/m_3.png)
![avg](./uses_image/m_4.png)
![avg](./uses_image/m_5.png)

`return x * weights.view(B, C, 1, 1)`
![avg](./uses_image/m_6.png)
![avg](./uses_image/m_7.png)
![avf](./uses_image/m_8.png)
![g](./uses_image/m_9.png)
![g](./uses_image/m_10.png)
![g](./uses_image/m_11.png)
![g](./uses_image/m_12.png)

![gkj](./uses_image/m_13.png)





`nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size,
    padding=padding,
    dilation=dilation
)
`
![a](./uses_image/b_1.png)
![v](./uses_image/b_2.png)