using UnityEngine;

public class ChainController : MonoBehaviour
{
    // Adjust these values based on your desired motor properties
    [SerializeField] private float desiredForce;
    [SerializeField] private float desiredTargetVelocity;
    [SerializeField] private float desiredDamper;

    void Start()
    {
        // Assuming the chain has 30 rings with Hinge Joint components
        for (int i = 1; i <= 30; i++)
        {
            // Construct the ring name based on your convention
            string ringName = "._" + i.ToString("D3");

            GameObject ring = GameObject.Find(ringName);
            if (ring != null)
            {
                AdjustHingeJointProperties(ring);
            }
            else
            {
                Debug.LogError("Ring not found: " + ringName);
            }
        }
    }

    void AdjustHingeJointProperties(GameObject ring)
    {
        HingeJoint hingeJoint = ring.GetComponent<HingeJoint>();

        if (hingeJoint != null)
        {
            hingeJoint.useMotor = true; // Enable the motor

            JointMotor motor = hingeJoint.motor;
            motor.force = desiredForce;
            motor.targetVelocity = desiredTargetVelocity;
            hingeJoint.motor = motor;

            JointSpring spring = hingeJoint.spring;
            spring.damper = desiredDamper;
            hingeJoint.spring = spring;
        }
        else
        {
            Debug.LogError("Hinge Joint not found on the ring: " + ring.name);
        }
    }
}