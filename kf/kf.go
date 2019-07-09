package main

import (
	"github.com/mrata/go-kalman-filter/ekf"
	"github.com/mrata/go-kalman-filter/linear"
	//"github.com/mrata/go-kalman-filter/ekf/whatever" // Importing a nested package
)

func main() {

	linear.KF_predict()
	linear.KF_update()

	ekf.EKF_predict()
	ekf.EKF_update()

}
