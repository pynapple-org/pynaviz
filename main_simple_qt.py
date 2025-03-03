"""
Test script
"""
import numpy as np
import pynapple as nap
import pynaviz as viz
from PyQt6.QtWidgets import QApplication

tsd1 = nap.Tsd(t=np.arange(1000), d=np.cos(np.arange(1000) * 0.1))
tsg = nap.TsGroup({
    i:nap.Ts(
        t=np.sort(np.random.uniform(0, 1000, 100*(i+1)))
    ) for i in range(10)})
tsdframe = nap.TsdFrame(t=np.arange(1000),d=np.random.randn(1000, 10))
tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))



app = QApplication([])

# viz.TsdWidget(tsd1).show()
viz.TsdTensorWidget(tsdtensor).show()



if __name__ == "__main__":
    app.exit(app.exec())
