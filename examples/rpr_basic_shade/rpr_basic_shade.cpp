/*
 * Vulkan Example - Texture loading (and display) example (including mip maps)
 *
 * Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gli/gli.hpp>

#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VulkanDevice.hpp"
#include "VulkanBuffer.hpp"

#include "Rpr/RadeonProRender.h"
#include "Rpr/RadeonProRender_Baikal.h"
#include "Rpr/RadeonProRender_VK.h"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION true

#ifndef WIN32
    #define PLUGIN_NAME "../libs/Rpr/Hybrid.so"
#else
    #define PLUGIN_NAME "../bin/Hybrid.dll"
#endif

#define CHECK_RPR(x) \
{ \
    rpr_int status = (x); \
    if (status != RPR_SUCCESS)    \
    { \
        std::cerr << "Error: " #x " == " << (status) << ")\n"; \
        assert(false);                      \
    }\
}

class VulkanExample : public VulkanExampleBase
{
public:
    // Contains all Vulkan objects that are required to store and use a texture
    // Note that this repository contains a texture class (VulkanTexture.hpp) that encapsulates texture loading functionality in a class that is used in subsequent demos
    struct Texture {
        VkSampler sampler;
        VkImage image;
        VkImageLayout imageLayout;
        VkImageView view;
    } color_aov, depth_aov;

    struct {
        VkPipelineVertexInputStateCreateInfo inputState;
        std::vector<VkVertexInputBindingDescription> bindingDescriptions;
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
    } vertices;

    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    uint32_t indexCount;

	struct Vertex {
        glm::vec4 position;
        glm::vec4 normal;
        glm::vec2 uv0;
        glm::vec2 uv1;
    };

    struct {
        VkPipeline solid;
		VkPipeline copy_depth;
        VkPipeline flat_shade;
    } pipelines;

    VkPipelineLayout pipelineLayout;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

	VkPipelineLayout copyDepthPipelineLayout;
    VkDescriptorSet copyDepthDescriptorSet;
    VkDescriptorSetLayout copyDepthDescriptorSetLayout;

	VkPipelineLayout flatShadePipelineLayout;
    VkDescriptorSetLayout flatShadeDescriptorSetLayout;

    //Interop
    constexpr static std::uint32_t frames_in_flight_ = 3;
    std::uint32_t acc_size_ = 2 * 1024u * 1024u;
    std::uint32_t vbuf_size_ = 1 * 1024u * 1024u;
    std::uint32_t ibuf_size_ = 1 * 1024u * 1024u;
    std::uint32_t sbuf_size_ = 5 * 1024u * 1024u;

    std::array<VkSemaphore, frames_in_flight_> framebuffer_release_semaphores_;
    std::array<VkSemaphore, frames_in_flight_> framebuffer_ready_semaphores_;

    rprContextFlushFrameBuffers_func rprContextFlushFrameBuffers;

    // RPR
    constexpr static std::size_t sphere_count_ = 5;
	struct object_data
	{
		rpr_shape shape = nullptr;
		rpr_material_node material = nullptr;
		VkBuffer index_buffer = nullptr;
		VkBuffer vertex_buffer = nullptr;
		std::uint64_t polygon_count = 0;
	};

    rpr_context context_ = nullptr;
    rpr_material_system mat_system_ = nullptr;
    rpr_scene scene_ = nullptr;
	std::vector<object_data> objects_;
    rpr_framebuffer color_framebuffer_ = nullptr;
	rpr_framebuffer depth_framebuffer_ = nullptr;
    rpr_camera camera_ = nullptr;

	struct
	{
		rpr_light light = nullptr;
		rpr_image image = nullptr;
	} env_light_;

    std::uint32_t semaphore_index_;
    std::int32_t quality = 0;
	std::int32_t shade_index_ = 0;

    VkPhysicalDeviceDescriptorIndexingFeaturesEXT desc_indexing;


    VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
    {
        zoom = -2.5f;
        rotation = { 0.0f, 15.0f, 0.0f };
        title = "RPR basic shade";
        settings.overlay = true;
        enabledDeviceExtensions.push_back(VK_EXT_SHADER_SUBGROUP_BALLOT_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);

        enabledFeatures.shaderInt64 = VK_TRUE;
        enabledFeatures.vertexPipelineStoresAndAtomics = VK_TRUE;
        enabledFeatures.fragmentStoresAndAtomics = VK_TRUE;
        enabledFeatures.geometryShader = VK_TRUE;
        enabledFeatures.independentBlend = VK_TRUE;

        memset(&desc_indexing, 0, sizeof(desc_indexing));
        desc_indexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
        desc_indexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;

        deviceCreatepNextChain = &desc_indexing;
    }

    ~VulkanExample()
    {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
		for (object_data& data : objects_)
		{
			ReleaseShapeMeshData(data);

			rprSceneDetachShape(scene_, data.shape);
			rprObjectDelete(data.shape);
			rprObjectDelete(data.material);
		}

		rprObjectDelete(env_light_.light);
		rprObjectDelete(env_light_.image);
        rprObjectDelete(scene_);
        rprObjectDelete(color_framebuffer_);
        rprObjectDelete(depth_framebuffer_);
        rprObjectDelete(mat_system_);
		rprObjectDelete(camera_);
        rprObjectDelete(context_);

        destroyTextureImage(color_aov);
		destroyTextureImage(depth_aov);

        vkDestroyPipeline(device, pipelines.solid, nullptr);
		vkDestroyPipeline(device, pipelines.copy_depth, nullptr);
		vkDestroyPipeline(device, pipelines.flat_shade, nullptr);

        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		vkDestroyPipelineLayout(device, copyDepthPipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, copyDepthDescriptorSetLayout, nullptr);

		vkDestroyPipelineLayout(device, flatShadePipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, flatShadeDescriptorSetLayout, nullptr);

		for (std::uint32_t i = 0; i < frames_in_flight_; ++i)
		{
			vkDestroySemaphore(device, framebuffer_release_semaphores_[i], nullptr);
		}

        vertexBuffer.destroy();
        indexBuffer.destroy();
    }

    // Enable physical device features required for this example
    virtual void getEnabledFeatures()
    {
        // Enable anisotropic filtering if supported
        if (deviceFeatures.samplerAnisotropy) {
            enabledFeatures.samplerAnisotropy = VK_TRUE;
        };
    }


    // Free all Vulkan resources used by a texture object
    void destroyTextureImage(Texture texture)
    {
        vkDestroyImageView(device, texture.view, nullptr);
        vkDestroySampler(device, texture.sampler, nullptr);
        vkDestroyImage(device, texture.image, nullptr);
    }

	void rebuildCommandBuffer(std::uint32_t const index)
    {
		VkCommandBuffer& cmdBuffer = drawCmdBuffers[index];

		vkResetCommandBuffer(cmdBuffer, 0);

        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

        VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

		{
			VkImageMemoryBarrier barrier = vks::initializers::imageMemoryBarrier();

			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
			barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			barrier.image = depthStencil.image;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = 1;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;

			vkCmdPipelineBarrier(
				cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0,
				0, nullptr, 0, nullptr, 1, &barrier
			);
		}

        VkClearValue clearValues[2];
        clearValues[0].color = defaultClearColor;
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.offset.x = 0;
        renderPassBeginInfo.renderArea.offset.y = 0;
        renderPassBeginInfo.renderArea.extent.width = width;
        renderPassBeginInfo.renderArea.extent.height = height;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        // Set target frame buffer
        renderPassBeginInfo.framebuffer = frameBuffers[index];

        vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
        vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

        VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
        vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.solid);

        vkCmdDraw(cmdBuffer, 3, 1, 0, 0);

		if (shade_index_ != 0)
		{
			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, copyDepthPipelineLayout, 0, 1, &copyDepthDescriptorSet, 0, NULL);
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.copy_depth);

			struct
			{
				glm::vec2 projectionCoefficients;
				float depthRange;
			} copyDepthConstants;

			glm::vec4 camera_pos;
			glm::vec3 aabb[2];
			CHECK_RPR(rprCameraGetInfo(camera_, RPR_CAMERA_POSITION, sizeof(glm::vec4), &camera_pos, nullptr));
			CHECK_RPR(rprSceneGetInfo(scene_, RPR_SCENE_AABB, sizeof(glm::vec3) * 2, aabb, nullptr));

			copyDepthConstants.projectionCoefficients = glm::vec2(cameraController.getProjection()[2][2], cameraController.getProjection()[3][2]);
			copyDepthConstants.depthRange = glm::length(glm::vec3(camera_pos) - (aabb[1] - aabb[0])) + glm::length(aabb[1] - aabb[0]) * 2;

			vkCmdPushConstants(cmdBuffer, copyDepthPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(copyDepthConstants), &copyDepthConstants);

			vkCmdDraw(cmdBuffer, 3, 1, 0, 0);


			object_data const& data = objects_[shade_index_];

			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.flat_shade);

			vkCmdBindIndexBuffer(cmdBuffer, data.index_buffer, 0, VK_INDEX_TYPE_UINT32);
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &data.vertex_buffer, &offset);

			struct
			{
				glm::mat4 transform;
				glm::vec4 color;
			} flatShadeConstants;

			flatShadeConstants.transform = cameraController.getProjection() * cameraController.getView();
			flatShadeConstants.color = glm::vec4(0.2f, 0.5f, 0.8f, 1.0f);

			vkCmdPushConstants(cmdBuffer, flatShadePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(flatShadeConstants), &flatShadeConstants);

			vkCmdDrawIndexed(cmdBuffer, data.polygon_count * 3, 1, 0, 0, 0);
		}

        drawUI(cmdBuffer);

        vkCmdEndRenderPass(cmdBuffer);

        VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
    }

    void draw()
    {
        VulkanExampleBase::prepareFrame();

		rebuildCommandBuffer(currentBuffer);

        std::array<VkSemaphore, 2> wait_semaphores =
        {
            semaphores.presentComplete,
            framebuffer_ready_semaphores_[semaphore_index_]
        };

        std::array<VkSemaphore, 2> signal_semaphores =
        {
            semaphores.renderComplete,
            framebuffer_release_semaphores_[semaphore_index_]
        };
        std::array<VkPipelineStageFlags, 2> wait_stages = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

        submitInfo.pWaitDstStageMask = wait_stages.data();

        // Command buffer to be sumitted to the queue
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

        submitInfo.waitSemaphoreCount = wait_semaphores.size();
        submitInfo.pWaitSemaphores = wait_semaphores.data();

        submitInfo.signalSemaphoreCount = signal_semaphores.size();
        submitInfo.pSignalSemaphores = signal_semaphores.data();

        // Submit to queue
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

        VulkanExampleBase::submitFrame();
    }

    void setupVertexDescriptions()
    {
        // Binding description
        vertices.inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
        vertices.inputState.vertexBindingDescriptionCount = 0;
        vertices.inputState.pVertexBindingDescriptions = nullptr;
        vertices.inputState.vertexAttributeDescriptionCount = 0;
        vertices.inputState.pVertexAttributeDescriptions = nullptr;
    }

    void setupDescriptorPool()
    {
        // Example uses one ubo and one image sampler
        std::vector<VkDescriptorPoolSize> poolSizes =
        {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2)
        };

        VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(
            static_cast<uint32_t>(poolSizes.size()),
            poolSizes.data(), 2);

        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
    }

    void setupDescriptorSetLayout()
    {
		{
			std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
			{
				// Binding 0 : Fragment shader image sampler
				vks::initializers::descriptorSetLayoutBinding(
					VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					VK_SHADER_STAGE_FRAGMENT_BIT,
					0)
			};

			VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));

			VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

			VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayout,
				1);

			VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayout));
		}

		{
			std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
			{
				vks::initializers::descriptorSetLayoutBinding(
					VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					VK_SHADER_STAGE_FRAGMENT_BIT,
					0
				)
			};

			VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));

			VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &copyDepthDescriptorSetLayout));

			VkPushConstantRange pushConstants =
				vks::initializers::pushConstantRange(
					VK_SHADER_STAGE_FRAGMENT_BIT,
					sizeof(glm::vec2) + sizeof(float),
					0);

			VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&copyDepthDescriptorSetLayout,
				1);

			pPipelineLayoutCreateInfo.pPushConstantRanges = &pushConstants;
			pPipelineLayoutCreateInfo.pushConstantRangeCount = 1;

			VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &copyDepthPipelineLayout));
		}

		{
			VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(nullptr, 0);

			VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &flatShadeDescriptorSetLayout));

			VkPushConstantRange pushConstants =
				vks::initializers::pushConstantRange(
					VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
					sizeof(glm::mat4) + sizeof(glm::vec4),
					0);

			VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&flatShadeDescriptorSetLayout,
				1);

			pPipelineLayoutCreateInfo.pPushConstantRanges = &pushConstants;
			pPipelineLayoutCreateInfo.pushConstantRangeCount = 1;

			VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &flatShadePipelineLayout));
		}
    }

    void setupDescriptorSet()
    {
		{
			VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayout,
				1);

			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
		}

		{
			VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&copyDepthDescriptorSetLayout,
				1);

			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &copyDepthDescriptorSet));
		}

		VkDescriptorImageInfo colorAovDescriptor;
		colorAovDescriptor.imageView = color_aov.view;
		colorAovDescriptor.sampler = color_aov.sampler;
		colorAovDescriptor.imageLayout = color_aov.imageLayout;

		VkDescriptorImageInfo depthAovDescriptor;
		depthAovDescriptor.imageView = depth_aov.view;
		depthAovDescriptor.sampler = depth_aov.sampler;
		depthAovDescriptor.imageLayout = depth_aov.imageLayout;

		std::vector<VkWriteDescriptorSet> writeDescriptorSets =
		{
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				0,
				&colorAovDescriptor
			),
			vks::initializers::writeDescriptorSet(
				copyDepthDescriptorSet,
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				0,
				&depthAovDescriptor
			)
		};

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
    }

    void preparePipelines()
    {
		{
			VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
				0,
				VK_FALSE);

			VkPipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				VK_POLYGON_MODE_FILL,
				VK_CULL_MODE_NONE,
				VK_FRONT_FACE_COUNTER_CLOCKWISE,
				0);

			VkPipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				0xf,
				VK_FALSE);

			VkPipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(
				1,
				&blendAttachmentState);

			VkPipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(
				VK_FALSE,
				VK_FALSE,
				VK_COMPARE_OP_ALWAYS);

			VkPipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

			VkPipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(
				VK_SAMPLE_COUNT_1_BIT,
				0);

			std::vector<VkDynamicState> dynamicStateEnables = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR
			};
			VkPipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables.data(),
				static_cast<uint32_t>(dynamicStateEnables.size()),
				0);

			// Load shaders
			std::array<VkPipelineShaderStageCreateInfo,2> shaderStages;

			shaderStages[0] = loadShader(getAssetPath() + "shaders/rpr_basic_shade/fullscreen_quad.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
			shaderStages[1] = loadShader(getAssetPath() + "shaders/rpr_basic_shade/output.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

			VkGraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayout,
				renderPass,
				0);

			pipelineCreateInfo.pVertexInputState = &vertices.inputState;
			pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
			pipelineCreateInfo.pRasterizationState = &rasterizationState;
			pipelineCreateInfo.pColorBlendState = &colorBlendState;
			pipelineCreateInfo.pMultisampleState = &multisampleState;
			pipelineCreateInfo.pViewportState = &viewportState;
			pipelineCreateInfo.pDepthStencilState = &depthStencilState;
			pipelineCreateInfo.pDynamicState = &dynamicState;
			pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
			pipelineCreateInfo.pStages = shaderStages.data();

			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.solid));
		}

		{
			VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
				0,
				VK_FALSE);

			VkPipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				VK_POLYGON_MODE_FILL,
				VK_CULL_MODE_NONE,
				VK_FRONT_FACE_COUNTER_CLOCKWISE,
				0);

			VkPipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				0x0,
				VK_FALSE);

			VkPipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(
				1,
				&blendAttachmentState);

			VkPipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(
				VK_TRUE,
				VK_TRUE,
				VK_COMPARE_OP_ALWAYS);

			VkPipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

			VkPipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(
				VK_SAMPLE_COUNT_1_BIT,
				0);

			std::vector<VkDynamicState> dynamicStateEnables = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR
			};
			VkPipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables.data(),
				static_cast<uint32_t>(dynamicStateEnables.size()),
				0);

			// Load shaders
			std::array<VkPipelineShaderStageCreateInfo,2> shaderStages;

			shaderStages[0] = loadShader(getAssetPath() + "shaders/rpr_basic_shade/fullscreen_quad.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
			shaderStages[1] = loadShader(getAssetPath() + "shaders/rpr_basic_shade/copy_depth.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

			VkGraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				copyDepthPipelineLayout,
				renderPass,
				0);

			pipelineCreateInfo.pVertexInputState = &vertices.inputState;
			pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
			pipelineCreateInfo.pRasterizationState = &rasterizationState;
			pipelineCreateInfo.pColorBlendState = &colorBlendState;
			pipelineCreateInfo.pMultisampleState = &multisampleState;
			pipelineCreateInfo.pViewportState = &viewportState;
			pipelineCreateInfo.pDepthStencilState = &depthStencilState;
			pipelineCreateInfo.pDynamicState = &dynamicState;
			pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
			pipelineCreateInfo.pStages = shaderStages.data();

			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.copy_depth));
		}

		{
			VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
				0,
				VK_FALSE);

			VkVertexInputAttributeDescription vertexInputAttributeDescription =
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0);

			VkVertexInputBindingDescription vertexInputBindingDescription =
			vks::initializers::vertexInputBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX);

			VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo =
			vks::initializers::pipelineVertexInputStateCreateInfo();
			vertexInputStateCreateInfo.vertexBindingDescriptionCount = 1;
			vertexInputStateCreateInfo.pVertexBindingDescriptions = &vertexInputBindingDescription;
			vertexInputStateCreateInfo.vertexAttributeDescriptionCount = 1;
			vertexInputStateCreateInfo.pVertexAttributeDescriptions = &vertexInputAttributeDescription;

			VkPipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				VK_POLYGON_MODE_FILL,
				VK_CULL_MODE_NONE,
				VK_FRONT_FACE_COUNTER_CLOCKWISE,
				0);
			rasterizationState.depthBiasEnable = VK_TRUE;
			rasterizationState.depthBiasConstantFactor = 1.0;
			rasterizationState.depthBiasSlopeFactor = 0.1;

			VkPipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				0xf,
				VK_FALSE);

			VkPipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(
				1,
				&blendAttachmentState);

			VkPipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(
				VK_TRUE,
				VK_FALSE,
				VK_COMPARE_OP_GREATER_OR_EQUAL);

			VkPipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

			VkPipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(
				VK_SAMPLE_COUNT_1_BIT,
				0);

			std::vector<VkDynamicState> dynamicStateEnables = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR
			};
			VkPipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables.data(),
				static_cast<uint32_t>(dynamicStateEnables.size()),
				0);

			// Load shaders
			std::array<VkPipelineShaderStageCreateInfo,2> shaderStages;

			shaderStages[0] = loadShader(getAssetPath() + "shaders/rpr_basic_shade/flat.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
			shaderStages[1] = loadShader(getAssetPath() + "shaders/rpr_basic_shade/flat.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

			VkGraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				flatShadePipelineLayout,
				renderPass,
				0);

			pipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo;
			pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
			pipelineCreateInfo.pRasterizationState = &rasterizationState;
			pipelineCreateInfo.pColorBlendState = &colorBlendState;
			pipelineCreateInfo.pMultisampleState = &multisampleState;
			pipelineCreateInfo.pViewportState = &viewportState;
			pipelineCreateInfo.pDepthStencilState = &depthStencilState;
			pipelineCreateInfo.pDynamicState = &dynamicState;
			pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
			pipelineCreateInfo.pStages = shaderStages.data();

			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.flat_shade));
		}
    }

    void initRpr()
    {
        //Register plugin
        rpr_int plugin_id = rprRegisterPlugin(PLUGIN_NAME);

        //Initialize rpr context with VK interop
        VkInteropInfo interop_info;
        VkInteropInfo::VkInstance instance;
        instance.device = device;
        instance.physical_device = physicalDevice;
        interop_info.instances = &instance;
        interop_info.instance_count = 1;
        interop_info.main_instance_index = 0;
        interop_info.frames_in_flight = frames_in_flight_;

        //Create framebuffer release semaphores required by RPR
        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (std::uint32_t a = 0; a < frames_in_flight_; ++a)
        {
            VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &framebuffer_release_semaphores_[a]));
        }
        interop_info.framebuffers_release_semaphores = framebuffer_release_semaphores_.data();

        // Set release semaphores to signalled state
        VkCommandBuffer fake_cmd_buffer;
        fake_cmd_buffer = createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VK_CHECK_RESULT(vkEndCommandBuffer(fake_cmd_buffer));

        VkSubmitInfo info;
        memset(&info, 0, sizeof(VkSubmitInfo));
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.pSignalSemaphores = framebuffer_release_semaphores_.data();
        info.signalSemaphoreCount = framebuffer_release_semaphores_.size();
        info.pWaitSemaphores = nullptr;
        info.waitSemaphoreCount = 0;
        info.pCommandBuffers = &fake_cmd_buffer;
        info.commandBufferCount = 1;


        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE));
        VK_CHECK_RESULT(vkQueueWaitIdle(queue));

        rpr_context_properties properties[] =
        {
            (void*)RPR_CONTEXT_CREATEPROP_VK_INTEROP_INFO, &interop_info,
            (void*)RPR_CONTEXT_CREATEPROP_HYBRID_ACC_MEMORY_SIZE, &acc_size_,
            (void*)RPR_CONTEXT_CREATEPROP_HYBRID_VERTEX_MEMORY_SIZE, &vbuf_size_,
            (void*)RPR_CONTEXT_CREATEPROP_HYBRID_INDEX_MEMORY_SIZE, &ibuf_size_,
            (void*)RPR_CONTEXT_CREATEPROP_HYBRID_STAGING_MEMORY_SIZE, &sbuf_size_,
            0
        };

        CHECK_RPR(rprCreateContext(RPR_API_VERSION, &plugin_id, 1,
            RPR_CREATION_FLAGS_ENABLE_GPU0 | RPR_CREATION_FLAGS_ENABLE_VK_INTEROP,
            properties, "cache", &context_));

        CHECK_RPR(rprContextSetParameterByKey1u(context_, RPR_CONTEXT_Y_FLIP, RPR_TRUE));

        //Get extension functions
        CHECK_RPR(rprContextGetFunctionPtr(context_, RPR_CONTEXT_FLUSH_FRAMEBUFFERS_FUNC_NAME, (void**)(&rprContextFlushFrameBuffers)));

        //Create material system
        CHECK_RPR(rprContextCreateMaterialSystem(context_, 0, &mat_system_));

        //Create scene
        CHECK_RPR(rprContextCreateScene(context_, &scene_))
    }

    void initAovs()
    {
		{
			rpr_framebuffer_desc desc = { (rpr_uint)width, (rpr_uint)height};
			rpr_framebuffer_format fmt = { 4, RPR_COMPONENT_TYPE_FLOAT16 };

			CHECK_RPR(rprContextCreateFrameBuffer(context_, fmt, &desc, &color_framebuffer_));
			CHECK_RPR(rprContextSetAOV(context_, RPR_AOV_COLOR, color_framebuffer_));
		}

		{
			rpr_framebuffer_desc desc = { (rpr_uint)width, (rpr_uint)height};
			rpr_framebuffer_format fmt = { 1, RPR_COMPONENT_TYPE_FLOAT32 };

			CHECK_RPR(rprContextCreateFrameBuffer(context_, fmt, &desc, &depth_framebuffer_));
			CHECK_RPR(rprContextSetAOV(context_, RPR_AOV_DEPTH, depth_framebuffer_));
		}

		//Get semaphores from color aov
		CHECK_RPR(rprContextGetInfo(context_, RPR_CONTEXT_FRAMEBUFFERS_READY_SEMAPHORES,
				sizeof(VkSemaphore) * frames_in_flight_, framebuffer_ready_semaphores_.data(), nullptr));
    }

    void initScene()
    {
        // Load up basic scene
        createScene();
        CHECK_RPR(rprContextSetScene(context_, scene_));

        //Init camera
        CHECK_RPR(rprContextCreateCamera(context_, &camera_));
        CHECK_RPR(rprCameraSetMode(camera_, RPR_CAMERA_MODE_PERSPECTIVE));

        glm::vec3 eye = glm::vec3(-0.2f, 1.3f, 12.6f);
        glm::vec3 up = glm::vec3(0.f, 1.f, 0.f);
        glm::vec3 at = glm::vec3(-0.2f, 1.3f, 5.6f);

        glm::vec2 sensor_size(0.036f, 0.024f); //Standart 36x24 sensor
		float focal_length = 0.035f;

        const float fovy = atan(sensor_size.y / (2.0f * focal_length));
        const float aspect = sensor_size.x / sensor_size.y;

        cameraController.setPerspective(fovy, aspect, 0.1f, 10000.f);
        cameraController.LookAt(eye, at, up);

        CHECK_RPR(rprCameraLookAt(camera_,
            eye.x, eye.y, eye.z,
            at.x, at.y, at.z,
            up.x, up.y, up.z));

        CHECK_RPR(rprCameraSetSensorSize(camera_, sensor_size.x * 1000.0, sensor_size.y * 1000.0));
        CHECK_RPR(rprSceneSetCamera(scene_, camera_));

        CHECK_RPR(rprContextCreateEnvironmentLight(context_, &env_light_.light));
        CHECK_RPR(rprContextCreateImageFromFile(context_, "../data/textures/hdr/studio015.hdr", &env_light_.image));
        CHECK_RPR(rprEnvironmentLightSetImage(env_light_.light, env_light_.image));
        CHECK_RPR(rprSceneSetEnvironmentLight(scene_, env_light_.light));

        CHECK_RPR(rprContextSetParameterByKey1f(context_, RPR_CONTEXT_DISPLAY_GAMMA, 2.2f));
    }

	void LoadShapeMeshData(object_data& data)
	{
		CHECK_RPR(rprMeshGetInfo(data.shape, RPR_MESH_VK_INDEX_BUFFER, sizeof(VkBuffer), &data.index_buffer, nullptr));
		CHECK_RPR(rprMeshGetInfo(data.shape, RPR_MESH_VK_VERTEX_BUFFER, sizeof(VkBuffer), &data.vertex_buffer, nullptr));
		CHECK_RPR(rprMeshGetInfo(data.shape, RPR_MESH_POLYGON_COUNT, sizeof(std::uint64_t), &data.polygon_count, nullptr));
	}

	void ReleaseShapeMeshData(object_data const& data)
	{
		if (data.index_buffer)
		{
			vkDestroyBuffer(device, data.index_buffer, nullptr);
		}

		if (data.vertex_buffer)
		{
			vkDestroyBuffer(device, data.vertex_buffer, nullptr);
		}
	}

    void CreateSphere(object_data& data, std::uint32_t lat, std::uint32_t lon, float radius, glm::vec3 const& center, glm::vec3 const& color)
    {
        size_t num_verts = (lat - 2) * lon + 2;
        size_t num_tris = (lat - 2) * (lon - 1) * 2;

        std::vector<glm::vec3> vertices(num_verts);
        std::vector<glm::vec3> normals(num_verts);
        std::vector<glm::vec2> uvs(num_verts);
        std::vector<std::uint32_t> indices(num_tris * 3);

        auto t = 0U;
        for (auto j = 1U; j < lat - 1; j++)
        {
            for (auto i = 0U; i < lon; i++)
            {
                float theta = float(j) / (lat - 1) * (float)M_PI;
                float phi = float(i) / (lon - 1) * (float)M_PI * 2;
                vertices[t].x = radius * sinf(theta) * cosf(phi) + center.x;
                vertices[t].y = radius * cosf(theta) + center.y;
                vertices[t].z = radius * -sinf(theta) * sinf(phi) + center.z;
                normals[t].x = sinf(theta) * cosf(phi);
                normals[t].y = cosf(theta);
                normals[t].z = -sinf(theta) * sinf(phi);
                uvs[t].x = phi / (2 * (float)M_PI);
                uvs[t].y = theta / ((float)M_PI);
                ++t;
            }
        }

        vertices[t].x = center.x; vertices[t].y = center.y + radius; vertices[t].z = center.z;
        normals[t].x = 0; normals[t].y = 1; normals[t].z = 0;
        uvs[t].x = 0; uvs[t].y = 0;
        ++t;
        vertices[t].x = center.x; vertices[t].y = center.y - radius; vertices[t].z = center.z;
        normals[t].x = 0; normals[t].y = -1; normals[t].z = 0;
        uvs[t].x = 1; uvs[t].y = 1;
        ++t;

        t = 0U;
        for (auto j = 0U; j < lat - 3; j++)
        {
            for (auto i = 0U; i < lon - 1; i++)
            {
                indices[t++] = j * lon + i;
                indices[t++] = (j + 1) * lon + i + 1;
                indices[t++] = j * lon + i + 1;
                indices[t++] = j * lon + i;
                indices[t++] = (j + 1) * lon + i;
                indices[t++] = (j + 1) * lon + i + 1;
            }
        }

        for (auto i = 0U; i < lon - 1; i++)
        {
            indices[t++] = (lat - 2) * lon;
            indices[t++] = i;
            indices[t++] = i + 1;
            indices[t++] = (lat - 2) * lon + 1;
            indices[t++] = (lat - 3) * lon + i + 1;
            indices[t++] = (lat - 3) * lon + i;
        }

        std::vector<int> faces(indices.size() / 3, 3);

        rpr_shape mesh = nullptr;
        CHECK_RPR(rprContextCreateMesh(context_,
            (rpr_float const*)vertices.data(), vertices.size(), sizeof(glm::vec3),
            (rpr_float const*)normals.data(), normals.size(), sizeof(glm::vec3),
            (rpr_float const*)uvs.data(), uvs.size(), sizeof(glm::vec2),
            (rpr_int const*)indices.data(), sizeof(rpr_int),
            (rpr_int const*)indices.data(), sizeof(rpr_int),
            (rpr_int const*)indices.data(), sizeof(rpr_int),
            faces.data(), faces.size(), &mesh));

		data.shape = mesh;
		LoadShapeMeshData(data);

        rpr_material_node material = nullptr;
        CHECK_RPR(rprMaterialSystemCreateNode(mat_system_, RPR_MATERIAL_NODE_UBERV2, &material));

        CHECK_RPR(rprMaterialNodeSetInputUByKey(material, RPR_UBER_MATERIAL_LAYERS, RPR_UBER_MATERIAL_LAYER_DIFFUSE));
        CHECK_RPR(rprMaterialNodeSetInputFByKey(material, RPR_MATERIAL_INPUT_UBER_DIFFUSE_COLOR, color.x, color.y, color.z, 0.0f));

        CHECK_RPR(rprShapeSetMaterial(mesh, material));

        CHECK_RPR(rprSceneAttachShape(scene_, mesh));

		data.material = material;
    }

    void CreatePlane(object_data& data, glm::vec3 center, glm::vec2 size, glm::vec3 normal, glm::vec3 const& color)
    {
        struct Vertex
        {
            glm::vec3 position;
            glm::vec3 normal;
            glm::vec2 uv;
        };

        glm::vec3 n = normalize(normal);
        glm::vec3 axis = fabs(n.x) > 0.001f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec3 t = normalize(cross(axis, n));
        glm::vec3 s = cross(n, t);

        Vertex vertices[4] =
        {
            { { -s * size.x - t * size.y + center }, n, { 0.0f, 0.0f } },
            { {  s * size.x - t * size.y + center }, n, { 1.0f, 0.0f } },
            { {  s * size.x + t * size.y + center }, n, { 1.0f, 1.0f } },
            { { -s * size.x + t * size.y + center }, n, { 0.0f, 1.0f } }
        };

        rpr_int indices[] =
        {
            3, 1, 0,
            2, 1, 3
        };

        rpr_int num_face_vertices[] =
        {
            3, 3
        };

        rpr_shape mesh = nullptr;

        CHECK_RPR(rprContextCreateMesh(context_,
            (rpr_float const*)&vertices[0], 4, sizeof(Vertex),
            (rpr_float const*)((char*)&vertices[0] + sizeof(glm::vec3)), 4, sizeof(Vertex),
            (rpr_float const*)((char*)&vertices[0] + sizeof(glm::vec3) * 2), 4, sizeof(Vertex),
            (rpr_int const*)indices, sizeof(rpr_int),
            (rpr_int const*)indices, sizeof(rpr_int),
            (rpr_int const*)indices, sizeof(rpr_int),
            num_face_vertices, 2, &mesh));

        rpr_material_node material = nullptr;
        CHECK_RPR(rprMaterialSystemCreateNode(mat_system_, RPR_MATERIAL_NODE_UBERV2, &material));

        CHECK_RPR(rprMaterialNodeSetInputUByKey(material, RPR_UBER_MATERIAL_LAYERS, RPR_UBER_MATERIAL_LAYER_DIFFUSE));
        CHECK_RPR(rprMaterialNodeSetInputFByKey(material, RPR_MATERIAL_INPUT_UBER_DIFFUSE_COLOR, color.x, color.y, color.z, 0.0f));

        CHECK_RPR(rprShapeSetMaterial(mesh, material));

        CHECK_RPR(rprSceneAttachShape(scene_, mesh));

		data.shape = mesh;
		data.material = material;
    }

    void createScene()
    {
		objects_.resize(6);

		CreatePlane(objects_[0], glm::vec3(0.0f, 0.0f, 0.0f), glm::vec2(10.0f, 10.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.4f, 0.4f, 0.4f));

        CreateSphere(objects_[1], 16, 16, 1.7f, glm::vec3(-6.0f, 2.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        CreateSphere(objects_[2], 16, 16, 1.7f, glm::vec3(-3.0f, 2.0f, 0.0f), glm::vec3(0.75f, 0.25f, 0.0f));
    	CreateSphere(objects_[3], 16, 16, 1.7f, glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(0.5f, 0.5f, 0.0f));
		CreateSphere(objects_[4], 16, 16, 1.7f, glm::vec3(3.0f, 2.0f, 0.0f), glm::vec3(0.25f, 0.75f, 0.0f));
		CreateSphere(objects_[5], 16, 16, 1.7f, glm::vec3(6.0f, 2.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	}

    void setupTextures()
    {
		{
			//get VkImage from FB
			CHECK_RPR(rprFrameBufferGetInfo(color_framebuffer_, RPR_VK_IMAGE_OBJECT, sizeof(VkImage), &color_aov.image, 0));

			// Create a texture sampler
			// In Vulkan textures are accessed by samplers
			// This separates all the sampling information from the texture data. This means you could have multiple sampler objects for the same texture with different settings
			// Note: Similar to the samplers available with OpenGL 3.3
			VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
			sampler.magFilter = VK_FILTER_NEAREST;
			sampler.minFilter = VK_FILTER_NEAREST;
			sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			sampler.mipLodBias = 0.0f;
			sampler.compareOp = VK_COMPARE_OP_NEVER;
			sampler.minLod = 0.0f;
			// Set max level-of-detail to mip level count of the texture
			sampler.maxLod = 0.0f;
			// Enable anisotropic filtering
			// This feature is optional, so we must check if it's supported on the device
			sampler.maxAnisotropy = 1.0;
			sampler.anisotropyEnable = VK_FALSE;
			sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
			VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &color_aov.sampler));

			// Create image view
			// Textures are not directly accessed by the shaders and
			// are abstracted by image views containing additional
			// information and sub resource ranges
			VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
			view.viewType = VK_IMAGE_VIEW_TYPE_2D;
			view.format = VK_FORMAT_R16G16B16A16_SFLOAT;
			view.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
			// The subresource range describes the set of mip levels (and array layers) that can be accessed through this image view
			// It's possible to create multiple image views for a single image referring to different (and/or overlapping) ranges of the image
			view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			view.subresourceRange.baseMipLevel = 0;
			view.subresourceRange.baseArrayLayer = 0;
			view.subresourceRange.layerCount = 1;
			// Linear tiling usually won't support mip maps
			// Only set mip map count if optimal tiling is used
			view.subresourceRange.levelCount = 1;
			// The view will be based on the texture's image
			view.image = color_aov.image;
			VK_CHECK_RESULT(vkCreateImageView(device, &view, nullptr, &color_aov.view));

			color_aov.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}

		{
			CHECK_RPR(rprFrameBufferGetInfo(depth_framebuffer_, RPR_VK_IMAGE_OBJECT, sizeof(VkImage), &depth_aov.image, 0));

			VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
			sampler.magFilter = VK_FILTER_NEAREST;
			sampler.minFilter = VK_FILTER_NEAREST;
			sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
			sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			sampler.mipLodBias = 0.0f;
			sampler.compareOp = VK_COMPARE_OP_NEVER;
			sampler.minLod = 0.0f;
			sampler.maxLod = 0.0f;
			sampler.maxAnisotropy = 1.0;
			sampler.anisotropyEnable = VK_FALSE;
			sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
			VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &depth_aov.sampler));

			VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
			view.viewType = VK_IMAGE_VIEW_TYPE_2D;
			view.format = VK_FORMAT_R32_SFLOAT;
			view.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
			view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			view.subresourceRange.baseMipLevel = 0;
			view.subresourceRange.baseArrayLayer = 0;
			view.subresourceRange.layerCount = 1;
			view.subresourceRange.levelCount = 1;
			view.image = depth_aov.image;
			VK_CHECK_RESULT(vkCreateImageView(device, &view, nullptr, &depth_aov.view));

			depth_aov.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
    }

    void prepare()
    {
        VulkanExampleBase::prepare();

        initRpr();
        initAovs();
        initScene();

		setupDepthStencil();
        setupTextures();

		setupFrameBuffer();
        setupVertexDescriptions();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    virtual void render()
    {
        if (!prepared)
            return;

        CHECK_RPR(rprContextRender(context_));
        CHECK_RPR(rprContextFlushFrameBuffers(context_));


        CHECK_RPR(rprContextGetInfo(context_, RPR_CONTEXT_INTEROP_SEMAPHORE_INDEX,
            sizeof(semaphore_index_), &semaphore_index_, nullptr));

        draw();
    }

    virtual void viewChanged()
    {
        auto pos = cameraController.camera->GetPosition();
        auto up = cameraController.camera->GetUpVector();
        auto at = cameraController.camera->GetAt();

        CHECK_RPR(rprCameraLookAt(camera_,
            pos.x, pos.y, pos.z, at.x, at.y, at.z, up.x, up.y, up.z));

        CHECK_RPR(rprFrameBufferClear(color_framebuffer_));
    }

    void updateQuality()
    {
        CHECK_RPR(rprContextSetParameterByKey1u(context_, RPR_CONTEXT_RENDER_QUALITY, quality));
        CHECK_RPR(rprFrameBufferClear(color_framebuffer_));
    }

    virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
    {
        if (overlay->header("Settings"))
        {
            if (overlay->comboBox("Quality", &quality, {"Low", "Medium", "High"}))
            {
                updateQuality();
            }

			overlay->comboBox("Shade ID", &shade_index_, {"None", "1", "2", "3", "4", "5"});
        }
    }
};

VULKAN_EXAMPLE_MAIN()
