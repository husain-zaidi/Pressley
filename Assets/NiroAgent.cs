using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.Robotics;
using Unity.Robotics.UrdfImporter.Control;

public class NiroAgent : Agent
{
    private ArticulationBody[] articulationChain;
    public Transform target;
    public Transform clothPos;

    // Start is called before the first frame update
    void Start()
    {
        articulationChain = this.GetComponentsInChildren<ArticulationBody>();
        foreach (ArticulationBody joint in articulationChain)
        {
            joint.gameObject.AddComponent<JointControl>();
            joint.jointFriction = 10;
            joint.angularDamping = 10;
            ArticulationDrive currentDrive = joint.xDrive;
            currentDrive.forceLimit = 1000;
            currentDrive.damping = 1000;
            currentDrive.stiffness = 100;
            joint.xDrive = currentDrive;
        }

    }

    public override void OnEpisodeBegin()
    {
        Debug.Log("Episode begin");
        // for all articulation body, set rotation 0
        foreach (ArticulationBody joint in articulationChain)
        {
            joint.gameObject.AddComponent<JointControl>();
            ArticulationDrive drive = joint.xDrive;
            drive.target = 0;
            joint.xDrive = drive;

            // force position
            float rotationRads = Mathf.Deg2Rad * 0;
            ArticulationReducedSpace newPosition = new ArticulationReducedSpace(rotationRads);
            joint.jointPosition = newPosition;

            // force velocity to zero
            ArticulationReducedSpace newVelocity = new ArticulationReducedSpace(0.0f);
            joint.jointVelocity = newVelocity;
        }
    }

    /// <summary>
    /// Move the robot arm
    /// </summary>
    /// <param name="actionBuffers"></param>
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Actions
        int i = 0;

        foreach (ArticulationBody joint in articulationChain)
        {
            JointControl current = joint.GetComponent<JointControl>();
            current.controltype = ControlType.PositionControl;

            float controlSignal = actionBuffers.ContinuousActions[i];
            if (controlSignal > 0)
            {
                current.direction = RotationDirection.Positive;
            }
            else if (controlSignal < 0)
            {
                current.direction = RotationDirection.Negative;
            }
            else
            {
                current.direction = RotationDirection.None;
            }
            i++;
        }
        // Rewards
        float distanceToTarget = Vector3.Distance(clothPos.localPosition, target.localPosition);
        if (distanceToTarget < 0.05f)
        {
            // Reached target
            SetReward(1.0f);
            EndEpisode();
        } 
        else if (distanceToTarget < 0.1f)
        {
            SetReward(1/distanceToTarget);
            Debug.Log("setting reward: "+ 1/distanceToTarget);
        }

        // // Fell off platform
        // else if (this.transform.localPosition.y < 0)
        // {
        //     EndEpisode();
        // }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
}
