import os
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List

import numpy as np
import traci
import pickle
from PIL import Image
from rich import print

from simModel.CarFactory import Vehicle, egoCar
from simModel.DataQueue import (
    CameraImages, ImageQueue, QAQueue, QuestionAndAnswer, RenderQueue,
)
from simModel.DBBridge import DBBridge
from simModel.MovingScene import MovingScene
from simModel.NetworkBuild import NetworkBuild
from utils.simBase import vehType
from utils.trajectory import Trajectory


class SettingErro(Exception):
    def __init__(self, errorInfo: str) -> None:
        super().__init__(self)
        self.errorInfo = errorInfo

    def __str__(self) -> str:
        return self.errorInfo


def resizeImage(img: np.ndarray, width: int, height: int) -> bytes:
    img = Image.fromarray(img)
    img_resized = img.resize((width, height))
    np_img_resized = np.array(img_resized)
    print(type(np_img_resized))
    return np_img_resized.tobytes()


class Model:
    def __init__(self, map_data ) -> None:
        print('[green bold]Model initialized at {}.[/green bold]'.format(
            datetime.now().strftime('%H:%M:%S.%f')[:-3]))

        self.sim_mode: str = 'RealTime'
        self.renderQueue = RenderQueue(5)
        self.imageQueue = ImageQueue(50)

        self.QAQ=None

        self.allvTypes = None

        position=map_data["map_point"]["position"]

        self.netBoundary = ((position[:,0].min(), position[:,1].min()), (position[:,0].max(), position[:,1].max()))


    def putRenderData(self):
        if self.tpStart:
            roadgraphRenderData, VRDDict = self.ms.exportRenderData()
            self.renderQueue.put((roadgraphRenderData, VRDDict))

    def exportSce(self):
        if self.tpStart:
            return self.ms.exportScene()
        else:
            return None, None

    def putCARLAImage(self):
        if self.CARLACosim:
            carla_ego = self.carlaSync.getEgo()
            if carla_ego:
                self.carlaSync.moveSpectator(carla_ego)
                self.carlaSync.setCameras(carla_ego)
                ci = self.carlaSync.getCameraImages()
                if ci:
                    ci.resizeImage(560, 315)
                    self.imageQueue.put(ci)
                    self.dbBridge.putData(
                        'imageINFO',
                        (
                            self.timeStep,
                            sqlite3.Binary(pickle.dumps(ci.CAM_FRONT)),
                            sqlite3.Binary(pickle.dumps(ci.CAM_FRONT_RIGHT)),
                            sqlite3.Binary(pickle.dumps(ci.CAM_FRONT_LEFT)),
                            sqlite3.Binary(pickle.dumps(ci.CAM_BACK_LEFT)),
                            sqlite3.Binary(pickle.dumps(ci.CAM_BACK)),
                            sqlite3.Binary(pickle.dumps(ci.CAM_BACK_RIGHT))
                        )
                    )
        else:
            return

    def getCARLAImage(
            self, start_frame: int, steps: int = 1
    ) -> List[CameraImages]:
        return self.imageQueue.get(start_frame, steps)

    def putQA(self, QA: QuestionAndAnswer):
        self.QAQ.put(QA)
        self.dbBridge.putData(
            'QAINFO',
            (
                self.timeStep, QA.description, QA.navigation,
                QA.actions, QA.few_shots, QA.response,
                QA.prompt_tokens, QA.completion_tokens, QA.total_tokens, QA.total_time, QA.choose_action
            )
        )

    def moveStep(self):
        traci.simulationStep()
        if self.CARLACosim:
            self.carlaSync.tick()
        self.timeStep += 1
        if self.ego.id in traci.vehicle.getIDList():
            self.getSce()
            self.putRenderData()
            self.putCARLAImage()
            if not self.tpStart:
                self.tpStart = 1

    def destroy(self):
        traci.close()
        self.dbBridge.close()
        if self.CARLACosim:
            self.carlaSync.close()
